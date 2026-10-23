#!/usr/bin/env python3
"""Seal GT-free full pair edges, source roles and reference index; never train.

This first stage opens only the sealed all-emitted index, its GT-free tracks,
full collection metadata and the fixed protocol. It does not open NPZ arrays,
manual targets, supervised rows, source GT, correctness outcomes or a model.
References remain compact: every observation of the sealed predicted track,
minus the two edge endpoints, in sampled-step order. All retained edges are
sealed before any later supervised subset may be joined.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "tmp_analysis/full_mean_pair_gate_protocol_20261023.json"
PROTOCOL_SHA = "fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718"
FULL_PLAN_SHA = "cbb14c6fd41135d887b46ce1b5869f0068b76922bf5c40f87f78ec2c2a715f9b"
ACCEPTED_MANIFEST = ROOT / "results/base_error_full_collection_20261023/support_v1/gt_free_index_manifest.json"
ACCEPTED_MANIFEST_SHA = "7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667"
ROLE_SEED = "pair_role_v1"
EDGE_SEED = "pair_edges_v1"
LAG_BINS = ((1, 2), (3, 5), (6, 10), (11, None))
CAP_PER_BIN = 4
MIN_TRACK = 4
OBS_FIELDS = {"observation_id", "video_id", "local_track_id", "image_id", "sampled_step",
              "original_frame_index", "split", "source_group", "dump_file", "dump_sha256",
              "dump_row", "pred_category_id", "score"}
OBS_REQUIRED = OBS_FIELDS - {"pred_category_id", "score"}
TRACK_FIELDS = {"video_id", "local_track_id", "split", "source_group", "n_observations", "observation_ids"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_manifest_binding(path, expected_path, expected_sha):
    require(Path(path).resolve() == expected_path, "Use the independently accepted GT-free manifest path")
    require(sha(path) == expected_sha, "Accepted GT-free manifest changed; independent re-review required")


def authenticate_manifest(path):
    """Bind the real entrypoint to the independently accepted upstream seal.

    A self-consistent replacement manifest is insufficient: its provenance
    must be the exact accepted full-support GT-free index, before any rows
    are parsed. There is deliberately no CLI override for either binding.
    """
    validate_manifest_binding(path, ACCEPTED_MANIFEST, ACCEPTED_MANIFEST_SHA)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def write(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            require(line.strip(), "Blank JSONL row: " + str(path))
            yield json.loads(line)


def assign_roles(video_records, selected_video_ids):
    """Assignment consumes only sealed source/split metadata, never observations."""
    groups = {}
    for vid in selected_video_ids:
        row = video_records[vid]
        group, split = str(row["source_group"]), row["split"]
        require(split in ("fit", "tune"), "Test video in full pair metadata")
        require(group not in groups or groups[group] == split, "Source group crosses split")
        groups[group] = split
    blocks = defaultdict(list)
    result = {}
    for group, split in groups.items():
        if split == "fit":
            blocks[group.split("/")[0]].append(group)
        else:
            result[group] = dict(split="tune", role="tune_evaluation")
    for block, members in sorted(blocks.items()):
        members.sort(key=lambda group: hashlib.sha256((ROLE_SEED + "\0" + group).encode()).hexdigest())
        for k, group in enumerate(members):
            result[group] = dict(split="fit", role="mean_role" if k % 2 == 0 else "pair_role", block=block)
    return dict(sorted(result.items()))


def lag_bin(lag):
    require(lag >= 1, "Nonpositive edge lag")
    for index, (low, high) in enumerate(LAG_BINS):
        if lag >= low and (high is None or lag <= high):
            return index
    raise ValueError("Uncovered lag")


def edge_digest(video, local, first, second):
    a, b = sorted((int(first), int(second)))
    require(a != b, "Self edge")
    value = "%s|%d|%d|%d|%d" % (EDGE_SEED, video, local, a, b)
    return hashlib.sha256(value.encode()).hexdigest()


def select_edges(video, local, observations):
    """Select hashes before target access; same hashes for swapped endpoints."""
    require(integer(video) and integer(local) and local >= 0, "Invalid video or local track ID")
    for row in observations:
        require(all(integer(row[k]) for k in ("observation_id", "image_id", "sampled_step", "original_frame_index")),
                "Noninteger track observation coordinate")
        require(row["observation_id"] >= 0 and row["sampled_step"] >= 0 and row["original_frame_index"] >= 0,
                "Negative track observation coordinate")
    ordered = sorted(observations, key=lambda row: (row["sampled_step"], row["observation_id"]))
    require(len({row["observation_id"] for row in ordered}) == len(ordered), "Duplicate track observation")
    require(len({row["sampled_step"] for row in ordered}) == len(ordered), "Duplicate local track in sampled frame")
    require(len({row["image_id"] for row in ordered}) == len(ordered), "Repeated image in one local track")
    require(all(a["original_frame_index"] < b["original_frame_index"] for a, b in zip(ordered, ordered[1:])),
            "Original frame indices must increase with sampled steps")
    if len(ordered) < MIN_TRACK:
        return [], [0] * len(LAG_BINS)
    heaps = [[] for _ in LAG_BINS]
    candidates = [0] * len(LAG_BINS)
    for first, second in itertools.combinations(ordered, 2):
        separation = second["sampled_step"] - first["sampled_step"]
        index = lag_bin(separation)
        candidates[index] += 1
        digest = edge_digest(video, local, first["observation_id"], second["observation_id"])
        # Max-heap by digest implemented using the negative integer. At most
        # four candidates per bin are retained, independent of track length.
        item = (-int(digest, 16), first["observation_id"], second["observation_id"], digest, first, second)
        if len(heaps[index]) < CAP_PER_BIN:
            heapq.heappush(heaps[index], item)
        elif item[0] > heaps[index][0][0]:
            heapq.heapreplace(heaps[index], item)
    all_ids = [row["observation_id"] for row in ordered]
    result = []
    for index, heap in enumerate(heaps):
        for _, _, _, digest, first, second in sorted(heap, key=lambda item: item[3]):
            a, b = first["observation_id"], second["observation_id"]
            refs = [oid for oid in all_ids if oid != a and oid != b]
            require(len(refs) == len(ordered) - 2 >= 2 and a not in refs and b not in refs, "Reference exclusion failed")
            result.append(dict(
                edge_id=digest, video_id=video, local_track_id=local,
                observation_i=a, observation_j=b,
                image_i=first["image_id"], image_j=second["image_id"],
                sampled_step_i=first["sampled_step"], sampled_step_j=second["sampled_step"],
                sample_step_lag=second["sampled_step"] - first["sampled_step"],
                original_frame_lag=second["original_frame_index"] - first["original_frame_index"],
                lag_bin=index, n_track_observations=len(ordered), n_reference_observations=len(refs),
                reference_observation_ids_sha256=json_hash(refs),
                disjoint_reference_supported=len(ordered) >= 6,
            ))
    return result, candidates


def metadata_plan(plan):
    require(plan["status"] == "SEALED_FULL_FIT_TUNE_METADATA_ONLY_NO_COLLECTION_RUN", "Wrong full plan status")
    records = {int(row["video_id"]): row for row in plan["video_records"]}
    require(len(records) == len(plan["video_records"]), "Duplicate planned video")
    images = {int(row["image_id"]): row for row in plan["images"]}
    require(len(images) == len(plan["images"]), "Duplicate planned image")
    selected_video_ids = {row["video_id"] for row in images.values()}
    require(all(records[vid]["split"] in ("fit", "tune") for vid in selected_video_ids), "Test frames selected")
    return records, images, assign_roles(records, selected_video_ids)


def load_index(index_manifest, plan, directory):
    """Project a strict GT-free schema; predicted classes/scores are never used."""
    require(index_manifest["status"] == "SEALED_GT_FREE_ALL_EMITTED_INDEX_BEFORE_SUPERVISION_OPEN", "Index was not sealed before supervision")
    require(index_manifest["aggregation_key"] == ["video_id", "local_track_id"], "Wrong local ID key")
    require(index_manifest["gt_filtering"] is False and index_manifest["supervision_opened"] is False,
            "GT-free seal contract failed")
    require(index_manifest["negative_local_ids_aggregated"] is False, "Negative IDs aggregated")
    require(index_manifest["posterior_category_ids"] == list(range(1, 1204)) + [-1], "Wrong full vocabulary")
    require(set(index_manifest["split_structure"]) <= {"fit", "tune"}, "Test index is forbidden for full pair preparation")
    require(set(index_manifest["files"]) == {"gt_free_observations.jsonl", "gt_free_tracks.jsonl"}, "Unexpected GT-free input file list")
    records, images, roles = metadata_plan(plan)
    grouped = defaultdict(list)
    seen = set()
    observation_counts = Counter()
    for row in jsonl(directory / "gt_free_observations.jsonl"):
        require(OBS_REQUIRED <= set(row) <= OBS_FIELDS, "Unexpected/missing observation field; possible supervision sidechannel")
        oid, vid, local, iid = row["observation_id"], row["video_id"], row["local_track_id"], row["image_id"]
        require(all(integer(v) for v in (oid, vid, local, iid, row["sampled_step"], row["original_frame_index"])), "Noninteger observation identity or time")
        require(oid >= 0 and oid not in seen and local >= 0, "Duplicate observation or unassigned local ID")
        seen.add(oid)
        require(iid in images and vid == images[iid]["video_id"], "Observation outside selected metadata")
        im, rec = images[iid], records[vid]
        require(row["split"] == rec["split"] in ("fit", "tune"), "Observation split mismatch/test")
        require(row["source_group"] == rec["source_group"], "Observation source-group mismatch")
        require(row["sampled_step"] == im["sample_step"] == im["frame_id"], "Sample-step provenance mismatch")
        require(row["original_frame_index"] == im["frame_index"], "Original frame provenance mismatch")
        require(Path(row["dump_file"]).name == row["dump_file"] and row["dump_file"].endswith(".npz"), "Invalid dump filename")
        require(integer(row["dump_row"]) and row["dump_row"] >= 0, "Invalid raw dump row")
        digest = row["dump_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef"), "Invalid sealed dump digest")
        grouped[(vid, local)].append({key: row[key] for key in (
            "observation_id", "image_id", "sampled_step", "original_frame_index")})
        observation_counts[row["split"]] += 1
    require(len(seen) == index_manifest["observations"], "Index observation total mismatch")
    for split, n in observation_counts.items():
        require(n == index_manifest["split_structure"][split]["emitted_rows"], "Split emitted total mismatch")
    listed_tracks = set()
    for row in jsonl(directory / "gt_free_tracks.jsonl"):
        require(set(row) == TRACK_FIELDS, "Unexpected track field; possible supervision sidechannel")
        require(integer(row["video_id"]) and integer(row["local_track_id"]) and integer(row["n_observations"]) and
                all(integer(oid) for oid in row["observation_ids"]), "Noninteger track identity or membership")
        key = (row["video_id"], row["local_track_id"])
        require(key in grouped and key not in listed_tracks, "Missing or duplicate GT-free track")
        listed_tracks.add(key)
        rec = records[key[0]]
        require(row["split"] == rec["split"] and row["source_group"] == rec["source_group"], "Track source/split mismatch")
        require(row["n_observations"] == len(grouped[key]) == len(row["observation_ids"]), "Track length mismatch")
        require(len(set(row["observation_ids"])) == len(row["observation_ids"]), "Duplicate track index observation")
        require(set(row["observation_ids"]) == {r["observation_id"] for r in grouped[key]}, "Track partition differs from observation index")
    require(listed_tracks == set(grouped) and len(grouped) == index_manifest["tracks"], "Track partition incomplete")
    return grouped, records, roles, observation_counts


def build(args):
    require(args.execute, "Real index sealing requires --execute after independent review")
    require(not args.out.exists(), "Use a new output directory")
    require(args.manifest.name == "gt_free_index_manifest.json", "Only sealed GT-free index manifest is supported")
    require(args.full_plan.name == "full_collection_plan.json", "Only full metadata plan is supported")
    authenticate_manifest(args.manifest)
    inputs = [PROTOCOL, args.manifest, args.full_plan,
              args.manifest.parent / "gt_free_observations.jsonl", args.manifest.parent / "gt_free_tracks.jsonl",
              Path(__file__).resolve()]
    before = {str(path): sha(path) for path in inputs}
    require(before[str(PROTOCOL)] == PROTOCOL_SHA, "Protocol changed; independent re-review required")
    require(before[str(args.full_plan)] == FULL_PLAN_SHA, "Full metadata selection changed")
    protocol = json.loads(PROTOCOL.read_text())
    edge_spec = protocol["pair_design"]["edge_protocol"]
    require(edge_spec["minimum_emitted_track_length"] == MIN_TRACK and edge_spec["per_track_cap_per_bin"] == CAP_PER_BIN,
            "Edge budget implementation disagrees with protocol")
    require(edge_spec["lag_bins_sample_steps"] == [list(v) for v in LAG_BINS], "Lag bins changed")
    manifest = json.loads(args.manifest.read_text())
    for name, digest in manifest["files"].items():
        require(name in {"gt_free_observations.jsonl", "gt_free_tracks.jsonl"}, "Unreviewed input path")
        require(before[str(args.manifest.parent / name)] == digest, "Sealed GT-free input changed: " + name)
    plan = json.loads(args.full_plan.read_text())
    started = time.monotonic()
    grouped, records, roles, observation_counts = load_index(manifest, plan, args.manifest.parent)
    args.out.mkdir(parents=True)
    write(args.out / "source_roles.json", dict(
        status="SEALED_METADATA_ONLY_SOURCE_ROLES", role_seed=ROLE_SEED,
        rule="Block by source-group prefix; hash order; alternate mean_role/pair_role; tune fixed evaluation",
        groups=roles, counts=dict(Counter(v["role"] for v in roles.values())), outcomes_used=False))
    counters = Counter()
    by_role = defaultdict(Counter)
    by_source = defaultdict(Counter)
    with (args.out / "track_contexts.jsonl").open("x") as track_file, (args.out / "edges.jsonl").open("x") as edge_file:
        for (vid, local), rows in sorted(grouped.items()):
            rec = records[vid]
            split, source = rec["split"], rec["source_group"]
            role = roles[source]["role"]
            rows.sort(key=lambda row: (row["sampled_step"], row["observation_id"]))
            # Validate duplicate local IDs even for unsupported short tracks.
            edges, candidate_counts = select_edges(vid, local, rows)
            counters["all_emitted_tracks"] += 1
            counters["all_emitted_observations"] += len(rows)
            if len(rows) < MIN_TRACK:
                counters["tracks_below_reference_support"] += 1
                continue
            track = dict(video_id=vid, local_track_id=local, split=split, source_group=source, role=role,
                         n_observations=len(rows), observation_ids=[r["observation_id"] for r in rows],
                         sampled_steps=[r["sampled_step"] for r in rows],
                         original_frame_indices=[r["original_frame_index"] for r in rows])
            track["observation_ids_sha256"] = json_hash(track["observation_ids"])
            track_file.write(json.dumps(track, separators=(",", ":"), ensure_ascii=False) + "\n")
            counters["supported_tracks"] += 1
            by_role[role]["supported_tracks"] += 1
            by_source[source]["supported_tracks"] += 1
            for index, n in enumerate(candidate_counts):
                counters["candidate_edges_lag_bin_%d" % index] += n
            for edge in edges:
                edge.update(split=split, source_group=source, role=role)
                edge_file.write(json.dumps(edge, separators=(",", ":"), ensure_ascii=False) + "\n")
                counters["retained_edges"] += 1
                counters["retained_edges_lag_bin_%d" % edge["lag_bin"]] += 1
                counters["edges_with_disjoint_reference_support"] += int(edge["disjoint_reference_supported"])
                by_role[role]["retained_edges"] += 1
                by_source[source]["retained_edges"] += 1
    require(counters["all_emitted_observations"] == manifest["observations"], "Dropped an emitted index row")
    after = {str(path): sha(path) for path in inputs}
    require(before == after, "Sealed input changed during pair preparation")
    files = {name: sha(args.out / name) for name in ("source_roles.json", "track_contexts.jsonl", "edges.jsonl")}
    result = dict(
        status="SEALED_GT_FREE_PAIR_EDGE_REFERENCE_INDEX_PENDING_INDEPENDENT_AUDIT",
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started, counters=dict(counters),
        split_observation_counts=dict(observation_counts), source_role_counts=dict(Counter(v["role"] for v in roles.values())),
        per_role={k: dict(v) for k, v in by_role.items()}, per_source={k: dict(v) for k, v in sorted(by_source.items())},
        source_groups_with_edges=len(by_source), input_sha256=before, input_sha256_after=after,
        opened_input_paths=[str(path) for path in inputs if path != Path(__file__).resolve()],
        files=files, protocol_sha256=PROTOCOL_SHA, full_collection_plan_sha256=FULL_PLAN_SHA,
        accepted_gt_free_manifest_sha256=ACCEPTED_MANIFEST_SHA,
        aggregation_key=["video_id", "local_track_id"],
        selection="GT-free all emitted tracks; max4 smallest edge hashes per sample-step lag bin; no target join",
        reference_definition="All track_contexts.observation_ids except observation_i/j, in stored sampled-step order; hash equals compact JSON array SHA256",
        supervision_opened=False, supervised_pair_filtering_performed=False, correctness_opened=False,
        test_opened=False, class_holdout_correctness_opened=False, posterior_or_feature_NPZ_opened=False,
        training_started=False, model_imported=False, GPU_used=False,
        downstream_note="This index must remain complete; later supervised joins only form another table, never replace GT-free edges/readout indices",
    )
    write(args.out / "pair_context_receipt.json", result)
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "counters", "source_role_counts", "source_groups_with_edges", "files")}, indent=2), flush=True)


def self_test():
    def rows(steps, base=0):
        return [dict(observation_id=base+k, image_id=10+base+k, sampled_step=s,
                     original_frame_index=s*30) for k, s in enumerate(steps)]
    r = rows([0, 1, 2, 4, 6, 8, 12, 16, 20, 25, 32])
    edges, candidates = select_edges(11, 7, r)
    require(len(edges) == 16, "Fixture should exercise cap in every lag bin")
    require(all(sum(e["lag_bin"] == b for e in edges) == CAP_PER_BIN for b in range(4)), "Per-bin cap failed")
    brute = defaultdict(list)
    for a, b in itertools.combinations(r, 2):
        brute[lag_bin(abs(a["sampled_step"] - b["sampled_step"]))].append(
            edge_digest(11, 7, a["observation_id"], b["observation_id"]))
    for b in range(4):
        require([e["edge_id"] for e in edges if e["lag_bin"] == b] == sorted(brute[b])[:CAP_PER_BIN], "Not minimum hashes")
    for e in edges:
        refs = [x["observation_id"] for x in r if x["observation_id"] not in (e["observation_i"], e["observation_j"])]
        require(json_hash(refs) == e["reference_observation_ids_sha256"] and len(refs) == len(r)-2, "Endpoint in reference")
        require(edge_digest(11, 7, e["observation_i"], e["observation_j"]) ==
                edge_digest(11, 7, e["observation_j"], e["observation_i"]), "Swap changed hash")
    shuffled = list(reversed(r))
    require(select_edges(11, 7, shuffled) == (edges, candidates), "Input order changed selection")
    poisoned = [dict(x, error=(k % 2), gt_category_id=999-k, score=k*.1,
                     pred_category_id=1203-k) for k, x in enumerate(r)]
    require(select_edges(11, 7, poisoned) == (edges, candidates), "Prediction/GT extras affected pure edge function")
    require(set(e["edge_id"] for e in edges).isdisjoint(e["edge_id"] for e in select_edges(12, 7, r)[0]), "Cross-video local ID collision")
    require(select_edges(11, 7, r[:3])[0] == [], "Insufficient references admitted")
    rejected = []
    for name, local, fixture in (("negative_local_id", -1, r), ("duplicate_sample_frame", 7, r+[dict(r[0], observation_id=999)]),
                                 ("duplicate_observation", 7, r+[dict(r[0], sampled_step=999)]),
                                 ("repeated_image", 7, [dict(x, image_id=10) for x in r]),
                                 ("decreasing_original_time", 7, [dict(x, original_frame_index=999-x["original_frame_index"]) for x in r]),
                                 ("bool_time", 7, [dict(r[0], sampled_step=False)]+r[1:])):
        try:
            select_edges(11, local, fixture)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("Accepted corrupt fixture: " + name)
    records = {k: dict(split="fit" if k < 6 else "tune", source_group="Dataset/g%d" % k) for k in range(8)}
    role_a = assign_roles(records, set(records))
    poisoned_records = {k: dict(v, error=1, gt_category_id=0) for k, v in records.items()}
    require(role_a == assign_roles(poisoned_records, set(records)), "GT changed source roles")
    require(Counter(v["role"] for v in role_a.values()) == {"mean_role":3, "pair_role":3, "tune_evaluation":2}, "Role block imbalance")
    for label, steps in (("one_to_two", [0,1,2]), ("three_to_five", [0,3,5]), ("six_to_ten", [0,6,10]), ("eleven_plus", [0,11,99])):
        # Boundary completeness checked independently from endpoint fixtures.
        require(all(0 <= lag_bin(b-a) <= 3 for a,b in itertools.combinations(steps,2)), label)
    with tempfile.TemporaryDirectory(prefix="pair_seal_auth_selftest_") as temporary:
        fixture = Path(temporary) / "gt_free_index_manifest.json"
        fixture.write_text('{"synthetic":true}\n')
        digest = sha(fixture)
        validate_manifest_binding(fixture, fixture.resolve(), digest)
        replacement = fixture.parent / "replacement" / fixture.name
        replacement.parent.mkdir()
        replacement.write_bytes(fixture.read_bytes())
        for label, candidate in (("replacement_manifest_path", replacement), ("changed_manifest_bytes", fixture)):
            if candidate == fixture:
                fixture.write_text('{"synthetic":true,"observations":1}\n')
            try:
                validate_manifest_binding(candidate, fixture.resolve(), digest)
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError("Accepted unauthenticated upstream fixture: " + label)
    return dict(status="PASS_CPU_GT_FREE_PAIR_INDEX_SYNTHETIC_ONLY", smallest_hash_per_bin=True,
                per_bin_cap=4, endpoint_swap_hash_invariant=True, input_order_invariant=True,
                no_endpoint_reference=True, GT_and_prediction_values_do_not_affect_pure_selection=True,
                cross_video_local_id_separate=True, role_assignment_metadata_only=True,
                upstream_manifest_path_and_hash_binding=True, original_time_strictly_increasing=True,
                below_four_has_no_edges=True, malformed_cases_rejected=rejected,
                data_or_posterior_opened=False, correctness_opened=False, training_started=False, GPU_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--full-plan", type=Path, default=ROOT / "results/base_error_full_collection_20261023/full_collection_plan.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        result["script_sha256"] = sha(__file__)
        result["protocol_sha256"] = sha(PROTOCOL)
        if args.self_test_output:
            require(not args.self_test_output.exists(), "Preserve synthetic receipt")
            write(args.self_test_output, result)
        print(json.dumps(result, indent=2), flush=True)
    else:
        require(args.manifest is not None and args.out is not None, "Provide --manifest and new --out")
        args.manifest, args.full_plan, args.out = args.manifest.resolve(), args.full_plan.resolve(), args.out.resolve()
        build(args)


if __name__ == "__main__":
    main()
