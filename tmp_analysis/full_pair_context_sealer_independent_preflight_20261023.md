# GT-free pair-context sealer independent preflight

PASS for implementation and CPU synthetic checks at script SHA `70ac6b7e844cbff3802ced02db732d04d32f72173e32e684ece0befdf9ddb654`, bound to frozen protocol SHA `fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718`. No real full index rows, NPZ arrays, supervision payloads or models were opened by these challenges.

Two implementation issues were corrected: the real entrypoint now accepts only the independently verified GT-free manifest path and SHA; malformed original timelines, repeated images and boolean integer aliases now fail closed. A structurally valid replacement index with matching declared file hashes is rejected by the fixed upstream binding. The frozen protocol and accepted support artifacts were preserved.

Three rounds each contained two checks: (1) path/hash authentication and input-access closure, (2) independent smallest-hash edge oracle over 53 synthetic track cases and independent blocked source roles, (3) strict schema/fault rejection and deterministic all-track-minus-two-endpoint references. Producer self-test and independent challenges exited 0. A second reader found no remaining blocker.

The four-edge bin cap limits output; selection still enumerates all track pairs. This review covers an index-only stage. Posterior pooling, disjoint-reference splitting, supervised joins, nuisance calibration and pair efficacy remain separate work. No mechanism or novelty claim follows.
