# Independent four-frame all-candidate benchmark completion audit

PASS: waited CPU worker exit 0 in 4.422678 seconds without timeout under 300 seconds. Protocol SHA `dfd211302b202e9678bb782e2543aae3501c5aa0eb118d370d24bddea1fd7ee6`, 18 input pins, and seven protected states matched before and after audit.

| Pair | candidates | donor fits | array bytes | peak RSS KiB | NPZ SHA |
|---|---:|---:|---:|---:|---|
| 1<-0 | 2700 | 2064 | 305100 | 48396 | `d4c03cf153c6ebf235eb37e91ef0ddc3e6e86befbc6fe67f9be86233ed49984f` |
| 2<-1 | 3024 | 2496 | 341712 | 48396 | `ff48c1f12a0f4dff05d838223ae26960f9ed6c9fe78ffd91aebecb009632d267` |
| 3<-2 | 2968 | 2288 | 335384 | 49152 | `40da36c3452105f2f62dc67e49f1b43f5e4cbb8ab284523db9606261ab500949` |

All 8,692 candidates are present and finite in this fixed run; the receipt-bound raw cosine is one canonical float64 full matrix per pair. Fit JSON provenance matches each candidate fit index and endpoint exclusion set. The audit did not rerun the slow oracle and did not read identity, GT, category, posterior, or released-ID values. No physical camera, identity, tracking, generalization, or novelty conclusion follows.

Receipt JSON SHA256: `0f3d302a28b47c46aca6c32f958754b9704b6bc5e57a5c26ff2adcf2ffffac70`.
