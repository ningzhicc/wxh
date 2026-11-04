# Four-dataset test evaluation

One fixed checkpoint; all frozen common_v1 test traces and all three start points.
Primary metrics exclude the first and last chunks. Startup stall is excluded.
Average start points within each trace, then average traces. SD uses ddof=0.

| Dataset | Traces | Sessions | Mean QoE | QoE SD | Bitrate (Mbps) | Stall (s/session) | Switch magnitude (Mbps) | Decision (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fcc | 100 | 300 | 1.105078 | 0.643315 | 1.202725 | 0.166298 | 0.082101 | 60.246 |
| norway3g | 41 | 123 | 1.366055 | 0.613929 | 1.617453 | 1.190137 | 0.140147 | 60.372 |
| oboe | 100 | 300 | 2.105263 | 1.088140 | 2.231609 | 0.115097 | 0.115587 | 60.507 |
| hsr | 34 | 102 | 2.891830 | 1.758925 | 3.499467 | 3.677691 | 0.263853 | 62.190 |

This report does not certify independence of the Merina656 training pool from the test traces.
