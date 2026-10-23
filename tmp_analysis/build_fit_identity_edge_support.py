#!/usr/bin/env python3
"""Describe FIT identity support on already sealed GT-free edges; never train.

The output is a separate diagnostic table. It never replaces the edge/readout
index, filters reference members, opens whole-track oracle purity, or measures
classification correctness. Missing eligible manual-anchor identity is UNKNOWN,
not a negative identity label. Even a two-anchor different-GT edge does not
establish the identity of its unlabelled references.

Real execution is disabled by default and requires independent implementation,
support-output and GT-free pair-output PASS receipts. This module uses only the
Python standard library; its self-test opens synthetic temporary files only.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/base_error_full_collection_20261023"
SUPPORT = STAGE / "support_v1"
CONTEXT = STAGE / "pair_context_v1"
TAG = "base_fit_identity_edge_support_20261023"
PREFLIGHT = ROOT / ("tmp_analysis/" + TAG + "_independent_preflight.json")
MANIFEST_SHA = "7006622770899f46de98132bb981f9519a69de80ef91cf6e533ba7ad79916667"
PAIR_AUDIT_SHA = "df7c386a8f01d3dbfce928e58758f9c455f685e8485dbfa8b19a0cd4ac936e81"
PAIR_RECEIPT_SHA = "ae9b5fd9a122533733e14208e81b0c875ca7bbb41ed1397ff4b804b25901dd2c"
SUPPORT_AUDIT_SHA = "92174aaa954f6e66cd0e7f520f2f883f7b9e0017c84fbf9bf4509e734ba714e8"
SUPPORT_RECEIPT_SHA = "e206051d0809b8e20b24529254eacece96f997c72b7155e8264197e7c07aae1e"
ROLES = ("mean_role", "pair_role")
AGREEMENTS = ("same_predicted_category", "different_predicted_category", "unavailable")
IDENTITIES = ("same_gt", "different_gt", "one_endpoint_unknown", "both_endpoints_unknown")
OBS_FIELDS = {
    "observation_id", "split", "source_group", "video_id", "image_id",
    "sampled_step", "original_frame_index", "local_track_id", "gt_track_id",
    "pred_category_id", "manual_anchor_id", "eligible_for_parameter_fit",
}
FORBIDDEN_FIELDS = {"error", "soft_risk", "gt_category_id", "posterior_argmax_category_id"}
NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def whitespace(text, pos):
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def string_end(text, pos):
    """Validate and skip JSON string bytes without decoding its value."""
    require(pos < len(text) and text[pos] == '"', "Expected JSON string")
    pos += 1
    while pos < len(text):
        c = text[pos]
        if c == '"':
            return pos + 1
        if c == "\\":
            pos += 1
            require(pos < len(text), "Truncated JSON escape")
            if text[pos] == "u":
                require(pos + 4 < len(text) and re.fullmatch(r"[0-9a-fA-F]{4}", text[pos+1:pos+5]),
                        "Invalid JSON Unicode escape")
                pos += 5
                continue
            require(text[pos] in '"\\/bfnrt', "Invalid JSON escape")
        else:
            require(ord(c) >= 32, "Unescaped JSON control character")
        pos += 1
    raise ValueError("Unterminated JSON string")


def value_end(text, pos, depth=0):
    """Lexically validate skipped values; never decode forbidden outcomes."""
    require(depth < 32 and pos < len(text), "Invalid JSON depth/end")
    c = text[pos]
    if c == '"':
        return string_end(text, pos)
    if c in "[{":
        close = "]" if c == "[" else "}"
        is_object = c == "{"
        pos = whitespace(text, pos + 1)
        if pos < len(text) and text[pos] == close:
            return pos + 1
        while True:
            if is_object:
                pos = whitespace(text, string_end(text, pos))
                require(pos < len(text) and text[pos] == ":", "Expected JSON colon")
                pos = whitespace(text, pos + 1)
            pos = whitespace(text, value_end(text, pos, depth + 1))
            require(pos < len(text), "Truncated JSON container")
            if text[pos] == close:
                return pos + 1
            require(text[pos] == ",", "Expected JSON comma")
            pos = whitespace(text, pos + 1)
    for token in ("true", "false", "null"):
        if text.startswith(token, pos):
            return pos + len(token)
    match = NUMBER.match(text, pos)
    require(match is not None, "Invalid JSON token")
    return match.end()


def project_object(text, wanted):
    """Decode only requested top-level fields, with duplicate-key rejection.

    Shared support JSONL contains tune and correctness bytes. These are streamed
    and hashed but never decoded as values. First decode only split; a tune row
    is discarded before any identity fields are requested. FIT rows then decode
    the small identity/coordinate/predicted-category allowlist only.
    """
    require(not (set(wanted) & FORBIDDEN_FIELDS), "Correctness projection forbidden")
    pos = whitespace(text, 0)
    require(pos < len(text) and text[pos] == "{", "Expected top-level object")
    pos = whitespace(text, pos + 1)
    result, seen = {}, set()
    if pos < len(text) and text[pos] == "}":
        pos += 1
    else:
        while True:
            end = string_end(text, pos)
            key = json.loads(text[pos:end])
            require(key not in seen, "Duplicate top-level JSON key: " + key)
            seen.add(key)
            pos = whitespace(text, end)
            require(pos < len(text) and text[pos] == ":", "Expected top-level colon")
            start = whitespace(text, pos + 1)
            end = value_end(text, start)
            if key in wanted:
                result[key] = json.loads(text[start:end])
            pos = whitespace(text, end)
            require(pos < len(text), "Truncated top-level object")
            if text[pos] == "}":
                pos += 1
                break
            require(text[pos] == ",", "Expected top-level comma")
            pos = whitespace(text, pos + 1)
    require(whitespace(text, pos) == len(text), "Trailing JSON data")
    return result


def projected_file(path, wanted):
    return project_object(Path(path).read_text(), wanted)


def load_fit_identities(path, roles):
    result = {}
    with Path(path).open() as handle:
        for line in handle:
            split = project_object(line, {"split"}).get("split")
            require(split in ("fit", "tune"), "Test/unknown support split forbidden")
            if split != "fit":
                continue  # no tune identity, category or correctness projection
            row = project_object(line, OBS_FIELDS)
            require(set(row) == OBS_FIELDS, "Missing FIT identity field")
            for field in OBS_FIELDS - {"source_group", "split", "eligible_for_parameter_fit"}:
                require(integer(row[field]) and row[field] >= 0, "Invalid integer identity/coordinate: " + field)
            require(row["pred_category_id"] > 0, "Invalid predicted category")
            group = row["source_group"]
            require(group in roles and roles[group]["split"] == "fit" and roles[group]["role"] in ROLES,
                    "FIT identity source missing from sealed role assignment")
            require(row["eligible_for_parameter_fit"] is True, "Ineligible FIT manual anchor")
            require(row["observation_id"] not in result, "Duplicate supervised observation ID")
            result[row["observation_id"]] = row
    return result


def edge_identity(edge, identities, roles):
    """One local endpoint join. No track-level identity state is consulted."""
    require(edge["split"] == "fit", "Only FIT edges may join supervision")
    group, role = edge["source_group"], edge["role"]
    require(group in roles and roles[group]["split"] == "fit" and roles[group]["role"] == role and role in ROLES,
            "Edge differs from sealed source role")
    for field in ("video_id", "local_track_id", "observation_i", "observation_j", "image_i", "image_j",
                  "sampled_step_i", "sampled_step_j", "sample_step_lag", "original_frame_lag", "lag_bin",
                  "n_track_observations", "n_reference_observations"):
        require(integer(edge[field]) and edge[field] >= 0, "Invalid edge integer: " + field)
    require(edge["observation_i"] != edge["observation_j"] and edge["image_i"] != edge["image_j"],
            "Repeated endpoint observation/image")
    lag = edge["sample_step_lag"]
    expected_bin = 0 if 1 <= lag <= 2 else 1 if 3 <= lag <= 5 else 2 if 6 <= lag <= 10 else 3 if lag >= 11 else -1
    require(edge["lag_bin"] == expected_bin and lag == edge["sampled_step_j"]-edge["sampled_step_i"],
            "Edge lag/bin mismatch")
    require(edge["original_frame_lag"] > 0 and edge["n_track_observations"] >= 4 and
            edge["n_reference_observations"] == edge["n_track_observations"]-2 >= 2,
            "Invalid all-emitted reference support")
    require(isinstance(edge["edge_id"], str) and re.fullmatch(r"[0-9a-f]{64}", edge["edge_id"]) and
            re.fullmatch(r"[0-9a-f]{64}", edge["reference_observation_ids_sha256"]), "Invalid edge/reference hash")
    endpoints = []
    for endpoint in ("i", "j"):
        row = identities.get(edge["observation_"+endpoint])
        if row is not None:
            require(row["split"] == "fit" and row["source_group"] == group and row["video_id"] == edge["video_id"] and
                    row["local_track_id"] == edge["local_track_id"] and row["image_id"] == edge["image_"+endpoint] and
                    row["sampled_step"] == edge["sampled_step_"+endpoint], "Illegal endpoint identity join")
        endpoints.append(row)
    a, b = endpoints
    if a is not None and b is not None:
        require(b["original_frame_index"]-a["original_frame_index"] == edge["original_frame_lag"],
                "Endpoint original timeline mismatch")
        label = "same_gt" if a["gt_track_id"] == b["gt_track_id"] else "different_gt"
        agreement = AGREEMENTS[int(a["pred_category_id"] != b["pred_category_id"])]
    else:
        label = "both_endpoints_unknown" if a is None and b is None else "one_endpoint_unknown"
        agreement = "unavailable"
    result = {key: edge[key] for key in (
        "edge_id", "video_id", "local_track_id", "source_group", "role", "observation_i", "observation_j",
        "sample_step_lag", "original_frame_lag", "lag_bin", "n_track_observations", "n_reference_observations",
        "reference_observation_ids_sha256",
    )}
    result.update(split="fit", identity_status=label, predicted_category_agreement=agreement,
                  gt_track_id_i=None if a is None else a["gt_track_id"],
                  gt_track_id_j=None if b is None else b["gt_track_id"],
                  manual_anchor_id_i=None if a is None else a["manual_anchor_id"],
                  manual_anchor_id_j=None if b is None else b["manual_anchor_id"])
    return result


def concentration(counter):
    values = sorted(counter.values(), reverse=True)
    n = sum(values)
    return dict(events=n, nonempty_units=len(values), max_unit_events=max(values) if values else 0,
                max_unit_share=values[0]/n if n else None,
                top5_share=sum(values[:5])/n if n else None, top10_share=sum(values[:10])/n if n else None,
                hhi=sum((v/n)**2 for v in values) if n else None,
                effective_units=n*n/sum(v*v for v in values) if n else None,
                events_after_deleting_largest_unit=n-(max(values) if values else 0))


class Cell:
    def __init__(self):
        self.n = 0
        self.events = Counter()
        self.sources = defaultdict(Counter)
        self.tracks = defaultdict(Counter)
        self.gt_pairs = defaultdict(Counter)
        self.trajectories = defaultdict(set)

    def add(self, row):
        label = row["identity_status"]
        source = row["source_group"]
        track = (row["video_id"], row["local_track_id"])
        self.n += 1
        self.events[label] += 1
        self.sources[label][source] += 1
        self.tracks[label][track] += 1
        if label in ("same_gt", "different_gt"):
            pair = (row["video_id"],) + tuple(sorted((row["gt_track_id_i"], row["gt_track_id_j"])))
            self.gt_pairs[label][pair] += 1
            self.trajectories[label].update((row["video_id"], t) for t in (row["gt_track_id_i"], row["gt_track_id_j"]))

    def result(self):
        positive = set(self.sources["same_gt"])
        negative = set(self.sources["different_gt"])
        return dict(retained_fit_edges=self.n, identity_counts={k:self.events[k] for k in IDENTITIES},
                    both_identity_known=self.events["same_gt"]+self.events["different_gt"],
                    sources_with_same_gt=len(positive), sources_with_different_gt=len(negative),
                    sources_with_both_identity_statuses=len(positive & negative),
                    tracks_with_both_identity_statuses=len(set(self.tracks["same_gt"]) & set(self.tracks["different_gt"])),
                    per_identity_concentration={label:dict(
                        source=concentration(self.sources[label]), predicted_track=concentration(self.tracks[label]),
                        official_gt_identity_pair=concentration(self.gt_pairs[label]),
                        official_gt_trajectories=len(self.trajectories[label]),
                        sources_with_ge_5_edges=sum(v >= 5 for v in self.sources[label].values()),
                        sources_with_ge_10_edges=sum(v >= 10 for v in self.sources[label].values()),
                    ) for label in IDENTITIES})


class Aggregates:
    def __init__(self, roles):
        self.all = Cell()
        self.role = {role:Cell() for role in ROLES}
        self.source = {g:Cell() for g, r in roles.items() if r["split"] == "fit"}
        self.marginal = {(role, lag, agree):Cell() for role in ROLES for lag in range(4) for agree in AGREEMENTS}
        self.source_cells = {}

    def add(self, row):
        key = (row["role"], row["lag_bin"], row["predicted_category_agreement"])
        sourcekey = (row["source_group"], row["lag_bin"], row["predicted_category_agreement"])
        if sourcekey not in self.source_cells:
            self.source_cells[sourcekey] = Cell()
        for cell in (self.all, self.role[row["role"]], self.source[row["source_group"]],
                     self.marginal[key], self.source_cells[sourcekey]):
            cell.add(row)

    def result(self, roles):
        return dict(all_fit=self.all.result(), per_role={k:v.result() for k,v in self.role.items()},
                    per_source={k:dict(role=roles[k]["role"], **v.result()) for k,v in sorted(self.source.items())},
                    role_lag_agreement=[dict(role=r, lag_bin=l, predicted_category_agreement=a, **v.result())
                                        for (r,l,a),v in sorted(self.marginal.items())],
                    source_lag_agreement=[dict(source_group=g, role=roles[g]["role"], lag_bin=l,
                        predicted_category_agreement=a, **v.result()) for (g,l,a),v in sorted(self.source_cells.items())])


def authenticate_inputs(preflight):
    """Open only receipt metadata/hash projections, never their metric fields."""
    require(Path(preflight).resolve() == PREFLIGHT, "Use the fixed independently reviewed implementation receipt")
    review = projected_file(preflight, {"status", "script_sha256"})
    require(review == dict(status="PASS_INDEPENDENT_FIT_IDENTITY_EDGE_SUPPORT_IMPLEMENTATION_PREFLIGHT",
                           script_sha256=sha(__file__)), "Identity implementation audit absent or stale")
    pair_audit_path = CONTEXT / "independent_pair_context_audit.json"
    require(sha(pair_audit_path) == PAIR_AUDIT_SHA, "Independently accepted pair output audit changed")
    pair_audit = projected_file(pair_audit_path, {"status", "pair_context_receipt_sha256", "output_files_sha256"})
    require(pair_audit["status"] == "PASS_INDEPENDENT_FULL_GT_FREE_PAIR_CONTEXT_AUDIT", "Pair output audit missing")
    pair_receipt_path = CONTEXT / "pair_context_receipt.json"
    require(sha(pair_receipt_path) == pair_audit["pair_context_receipt_sha256"] == PAIR_RECEIPT_SHA, "Pair receipt changed after audit")
    pair = projected_file(pair_receipt_path, {"status", "files", "accepted_gt_free_manifest_sha256",
        "supervision_opened", "supervised_pair_filtering_performed", "correctness_opened", "test_opened",
        "class_holdout_correctness_opened", "posterior_or_feature_NPZ_opened", "training_started"})
    require(pair["status"] == "SEALED_GT_FREE_PAIR_EDGE_REFERENCE_INDEX_PENDING_INDEPENDENT_AUDIT", "Wrong pair seal")
    require(pair["accepted_gt_free_manifest_sha256"] == MANIFEST_SHA, "Wrong upstream GT-free manifest")
    for field in ("supervision_opened", "supervised_pair_filtering_performed", "correctness_opened", "test_opened",
                  "class_holdout_correctness_opened", "posterior_or_feature_NPZ_opened", "training_started"):
        require(pair[field] is False, "Pair boundary violated: " + field)
    require(pair["files"] == pair_audit["output_files_sha256"] and
            set(pair["files"]) == {"source_roles.json", "track_contexts.jsonl", "edges.jsonl"}, "Pair audited outputs differ")
    support_audit_path = SUPPORT / "independent_support_audit.json"
    require(sha(support_audit_path) == SUPPORT_AUDIT_SHA, "Independently accepted support output audit changed")
    audited = projected_file(support_audit_path, {"status", "support_receipt_sha256", "gt_free_index_manifest_sha256"})
    require(audited["status"] == "PASS_INDEPENDENT_FULL_SUPPORT_OUTPUT_AUDIT", "Support output audit missing")
    support_receipt_path = SUPPORT / "support_receipt.json"
    require(sha(support_receipt_path) == audited["support_receipt_sha256"] == SUPPORT_RECEIPT_SHA, "Support receipt changed after audit")
    support = projected_file(support_receipt_path, {"status", "output_files", "gt_free_index_manifest_sha256",
        "unmatched_is_unknown", "class_holdout_outcomes_computed", "test_outcomes_computed", "training_started",
        "oracle_identity_diagnostics_used_for_training", "collection_scope"})
    require(support["status"] == "PASS_FIT_TUNE_SIGNAL_SUPPORT_AUDIT" and
            support["collection_scope"] == "FULL_FIT_TUNE_EXACT_MANUAL_ANCHOR_FRAMES", "Wrong support scope")
    require(support["unmatched_is_unknown"] is True and all(support[k] is False for k in (
        "class_holdout_outcomes_computed", "test_outcomes_computed", "training_started", "oracle_identity_diagnostics_used_for_training")),
        "Support boundary violation")
    manifest_path = SUPPORT / "gt_free_index_manifest.json"
    require(sha(manifest_path) == audited["gt_free_index_manifest_sha256"] == support["gt_free_index_manifest_sha256"] == MANIFEST_SHA,
            "Accepted GT-free manifest changed")
    inputs = {str(p):sha(p) for p in (
        Path(__file__).resolve(), Path(preflight), pair_audit_path, pair_receipt_path,
        support_audit_path, support_receipt_path, manifest_path, SUPPORT / "supervised_observations.jsonl",
        CONTEXT / "source_roles.json", CONTEXT / "edges.jsonl", CONTEXT / "track_contexts.jsonl")}
    require(inputs[str(SUPPORT/"supervised_observations.jsonl")] == support["output_files"]["supervised_observations.jsonl"],
            "Supervised row file changed after support audit")
    for name, digest in pair["files"].items():
        require(inputs[str(CONTEXT/name)] == digest, "Pair output changed after audit: " + name)
    return inputs


def build(args):
    require(args.execute, "Real FIT identity join requires --execute after independent review")
    out = args.out.resolve()
    require(out.parent == STAGE and out.name == TAG and not out.exists(), "Use the new fixed identity-diagnostic output directory")
    started = time.monotonic()
    before = authenticate_inputs(args.implementation_audit)
    role_document = json.loads((CONTEXT/"source_roles.json").read_text())
    require(role_document["status"] == "SEALED_METADATA_ONLY_SOURCE_ROLES" and role_document["outcomes_used"] is False,
            "Source roles were not sealed without outcomes")
    roles = role_document["groups"]
    require(all(r["split"] in ("fit", "tune") for r in roles.values()) and
            all(r["role"] in ROLES for r in roles.values() if r["split"] == "fit"), "Invalid sealed role scope")
    identities = load_fit_identities(SUPPORT/"supervised_observations.jsonl", roles)
    aggregates = Aggregates(roles)
    seen = set()
    out.mkdir()
    with (CONTEXT/"edges.jsonl").open() as source, (out/"fit_edge_identity_rows.jsonl").open("x") as target:
        for line in source:
            edge = json.loads(line)  # this file is independently audited GT-free metadata
            require(edge["split"] in ("fit", "tune"), "Test/unknown edge forbidden")
            if edge["split"] != "fit":
                continue  # no tune endpoint join or tune statistics
            require(edge["edge_id"] not in seen, "Duplicate FIT edge")
            seen.add(edge["edge_id"])
            row = edge_identity(edge, identities, roles)
            aggregates.add(row)
            target.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n")
    stats = aggregates.result(roles)
    require(stats["all_fit"]["retained_fit_edges"] == len(seen) and
            sum(v["retained_fit_edges"] for v in stats["per_role"].values()) == len(seen), "Dropped FIT selected edge")
    require(before == {path:sha(path) for path in before}, "Protected input changed during FIT identity join")
    summary = dict(status="COMPLETE_FIT_IDENTITY_EDGE_STRUCTURE_DIAGNOSTIC_PENDING_INDEPENDENT_AUDIT",
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started, fit_eligible_manual_identity_observations=len(identities),
        fit_role_source_counts=dict(Counter(r["role"] for r in roles.values() if r["split"] == "fit")),
        support=stats, correctness_fields_decoded=[], fit_identity_fields_decoded=sorted(OBS_FIELDS),
        fit_tune_shared_bytes_policy="Mixed support bytes hashed/streamed; only split decoded on tune rows; FIT identity allowlist only. No classification correctness value decoded.",
        tune_identity_joined=False, tune_results_reopened=False, test_opened=False, class_holdout_correctness_opened=False,
        classification_correctness_computed=False, training_started=False, model_imported=False, GPU_used=False,
        selected_edges_replaced=False, all_emitted_references_changed=False, whole_track_purity_used=False,
        unknown_is_negative=False, pair_training=False, new_hypothesis_efficacy=False, training_readiness_established=False,
        identity_scope="Eligible exact manual-anchor endpoint identities only; same GT versus different GT within each sealed predicted track",
        interpretation=["Descriptive FIT identity support only; this is NOT pair training or new hypothesis efficacy.",
            "Different-GT edges are observed endpoint identity contamination, not class disagreement or classification errors.",
            "Unknown endpoints include unmatched, nonmanual, heldout-class, ignored and ambiguous observations; no individual reason is inferred.",
            "Agreement is emitted predicted-category equality only when both endpoints have eligible identity rows; unavailable otherwise.",
            "All selected FIT edges remain represented; no whole-track purity filter or GT reference selection is applied.",
            "Pair counts and effective source counts are descriptive concentration summaries, not independent sample counts or statistical power.",
            "Sparse manual labels do not establish whole-track/reference purity or the prevalence of all contamination.",
            "No support-sufficiency, novelty, efficacy, retraining or deployable identity-separation gate is passed by this table."],
        input_sha256=before, all_input_hashes_verified_before_after=True,
        output_files_sha256={"fit_edge_identity_rows.jsonl":sha(out/"fit_edge_identity_rows.jsonl")})
    write(out/"identity_support_summary.json", summary)
    print(json.dumps({"status":summary["status"], "summary":str(out/"identity_support_summary.json"),
                      "summary_sha256":sha(out/"identity_support_summary.json"),
                      "fit_retained_edges":len(seen), "elapsed_seconds":summary["elapsed_seconds"]}, indent=2))


def self_test():
    roles = {"S/A":dict(split="fit",role="mean_role"), "S/B":dict(split="fit",role="pair_role"),
             "S/empty":dict(split="fit",role="mean_role"), "S/tune":dict(split="tune",role="tune_evaluation")}
    def observation(oid, gt, pred=1, group="S/A", video=10, local=7):
        return dict(observation_id=oid, split="fit",source_group=group,video_id=video,image_id=100+oid,
            sampled_step=oid,original_frame_index=oid*3,local_track_id=local,gt_track_id=gt,pred_category_id=pred,
            manual_anchor_id=200+oid,eligible_for_parameter_fit=True)
    observations = [observation(1,5),observation(2,5),observation(3,6,2),observation(4,6),
                    observation(11,5,group="S/B",video=11),observation(12,8,group="S/B",video=11)]
    poison = dict(split="tune", source_group="S/tune", gt_track_id={"poison":[1,2]},error={"poison":[True]},
                  soft_risk="not a risk",gt_category_id="CLOSED")
    with tempfile.TemporaryDirectory(prefix="fit_identity_support_selftest_") as temporary:
        path=Path(temporary)/"supervised_observations.jsonl"
        with path.open("w") as handle:
            for row in observations+[poison]:
                row=dict(row,error={"DO_NOT_DECODE":["\\\"{}[]",None]},soft_risk=[999],gt_category_id="POISON")
                handle.write(json.dumps(row)+"\n")
        original_loads=json.loads
        def guarded_loads(text,*args,**kwargs):
            require(not any(token in text for token in ("DO_NOT_DECODE","POISON","CLOSED","S/tune")),
                    "Forbidden value was passed to a JSON decoder")
            return original_loads(text,*args,**kwargs)
        json.loads=guarded_loads
        try:
            identities=load_fit_identities(path,roles)
        finally:
            json.loads=original_loads
        require(len(identities)==6 and all(not(FORBIDDEN_FIELDS & set(r)) for r in identities.values()), "Projection leaked outcomes")
        def edge(a,b,group="S/A",video=10):
            lag=b-a
            return dict(edge_id=hashlib.sha256((group+str(a)+","+str(b)).encode()).hexdigest(),split="fit",
                source_group=group,role=roles[group]["role"],video_id=video,local_track_id=7,
                observation_i=a,observation_j=b,image_i=100+a,image_j=100+b,sampled_step_i=a,sampled_step_j=b,
                sample_step_lag=lag,original_frame_lag=lag*3,lag_bin=0 if lag<=2 else 1 if lag<=5 else 2 if lag<=10 else 3,
                n_track_observations=20,n_reference_observations=18,reference_observation_ids_sha256="a"*64)
        fixture=[edge(1,2),edge(1,3),edge(1,4),edge(1,9),edge(8,9),edge(11,12,"S/B",11)]
        rows=[edge_identity(e,identities,roles) for e in fixture]
        require([r["identity_status"] for r in rows]==["same_gt","different_gt","different_gt","one_endpoint_unknown","both_endpoints_unknown","different_gt"], "Endpoint identity labels differ")
        require([r["predicted_category_agreement"] for r in rows[:3]]==[AGREEMENTS[0],AGREEMENTS[1],AGREEMENTS[0]], "Agreement differs")
        aggregates=Aggregates(roles)
        for row in rows: aggregates.add(row)
        stats=aggregates.result(roles)
        require(stats["all_fit"]["identity_counts"]==dict(same_gt=1,different_gt=3,one_endpoint_unknown=1,both_endpoints_unknown=1), "Unknown became negative/dropped")
        require(stats["per_source"]["S/empty"]["retained_fit_edges"]==0 and len(stats["role_lag_agreement"])==24, "Missing empty source/cell")
        c=stats["all_fit"]["per_identity_concentration"]["different_gt"]["source"]
        require(c["events"]==3 and c["max_unit_share"]==2/3 and abs(c["effective_units"]-1.8)<1e-12 and c["events_after_deleting_largest_unit"]==1, "Concentration differs")
        # Adding other identities on the same predicted track cannot filter an edge.
        mutated=dict(identities); mutated[90]=observation(90,999)
        require(rows==[edge_identity(e,mutated,roles) for e in fixture], "Whole-track identity affected local join")
        require(all(r["reference_observation_ids_sha256"]==e["reference_observation_ids_sha256"] and
                    r["n_reference_observations"]==e["n_reference_observations"] for r,e in zip(rows,fixture)), "References changed")
        rejected=[]
        for name,changed in [("test_edge",dict(fixture[0],split="test")),
                ("role_changed",dict(fixture[0],role="pair_role")),
                ("endpoint_video_changed",dict(fixture[0],video_id=11)),
                ("endpoint_local_changed",dict(fixture[0],local_track_id=8)),
                ("lag_changed",dict(fixture[0],lag_bin=3)),
                ("references_changed",dict(fixture[0],n_reference_observations=17))]:
            try: edge_identity(changed,identities,roles)
            except ValueError: rejected.append(name)
            else: raise AssertionError("Accepted mutation: "+name)
        for name,text,wanted in [("duplicate_split",'{"split":"fit","split":"tune"}',{"split"}),
                ("forbidden_projection",'{"error":1}',{"error"}), ("trailing_data",'{"split":"fit"}{}',{"split"}),
                ("invalid_literal",'{"split":"fit","error":NaN}',{"split"})]:
            try: project_object(text,wanted)
            except ValueError: rejected.append(name)
            else: raise AssertionError("Accepted projection mutation: "+name)
        path.write_text(json.dumps(dict(observations[0],split="test"))+"\n")
        try: load_fit_identities(path,roles)
        except ValueError: rejected.append("test_support_row")
        else: raise AssertionError("Accepted test supervision")
        path.write_text(json.dumps(observations[0])+"\n"+json.dumps(observations[0])+"\n")
        try: load_fit_identities(path,roles)
        except ValueError: rejected.append("duplicate_supervision")
        else: raise AssertionError("Accepted duplicate supervision")
    return dict(status="PASS_SYNTHETIC_FIT_IDENTITY_EDGE_STRUCTURE_SUPPORT",script_sha256=sha(__file__),
        synthetic_selected_edges=6,identity_counts=stats["all_fit"]["identity_counts"],
        tune_poison_skipped_before_identity_projection=True,classification_outcome_values_not_decoded=True,
        unknown_not_negative=True,whole_track_purity_not_used=True,reference_hashes_unchanged=True,
        empty_sources_and_fixed_cells_preserved=True,source_concentration_verified=True,rejected_mutations=rejected,
        real_support_opened=False,real_pair_edges_opened=False,training_started=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--self-test-out",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--implementation-audit",type=Path,default=PREFLIGHT)
    parser.add_argument("--out",type=Path,default=STAGE/TAG)
    args=parser.parse_args()
    require(not(args.self_test and args.execute), "Synthetic and real modes are exclusive")
    if args.self_test:
        result=self_test()
        if args.self_test_out is not None: write(args.self_test_out,result)
        print(json.dumps(result,indent=2))
    else:
        build(args)


if __name__=="__main__":
    main()
