# Independent four-frame ROI observation completion audit

PASS. Actual outer exit0, no timeout; elapsed25.916793s under recorded300s limit. All24 protocol input pins,71 original model source hashes, and7 protected existence/hash states independently matched before and after audit. Exactly4 NPZ output SHAs match the observation receipt.

| Frame | N_pre | Retained | Spatial ROI/conv shape | Raw shape |
|---|---:|---:|---|---|
| v1271 f0 | 69 | 50 | [50, 256, 7, 7] | [50, 256] |
| v1271 f1 | 80 | 54 | [54, 256, 7, 7] | [54, 256] |
| v801 f0 | 80 | 69 | [69, 256, 7, 7] | [69, 256] |
| v801 f1 | 80 | 68 | [68, 256, 7, 7] | [68, 256] |

All241 retained rows are finite float32 NCHW. Saved pretracker rows equal nonzero original keep mask; proposal rows and original-space xyxy match. Independent saved raw→float16 uint16-bit comparison equals the parent raw cache.

Runtime evidence is native full-N_pre spatial→FC→embed exact replay, full N_pre model-space geometry, live keep/reset controls, followed by full raw CPU NumPy→float16→keep cache equality. This final audit verified the frozen code/receipt/actualexit chain; it did not rerun the model or FC.

The separate FC replay of saved retained spatial tokens remains unverified. Four frames establish bounded observability/alignment only; no physical visibility, identified motion, identity/tracking gain, generalization or novelty follows. Original historical runtime flags and input hashes remain incompletely sealed as documented in protocol.

Receipt SHA256: `e3400041954ca9cfdaf51632a14c1eb07904dfc8c52d311534087b5ac5759a95`.
