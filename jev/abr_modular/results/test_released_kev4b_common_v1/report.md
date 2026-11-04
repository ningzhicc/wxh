# Four-dataset test evaluation

One fixed checkpoint; all frozen common_v1 test traces and all three start points.
Primary metrics exclude the first and last chunks. Startup stall is excluded.
Average start points within each trace, then average traces. SD uses ddof=0.

| Dataset | Traces | Sessions | Mean QoE | QoE SD | Bitrate (Mbps) | Stall (s/session) | Switch magnitude (Mbps) | Decision (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fcc | 100 | 300 | -33.141723 | 28.709398 | 2.885304 | 384.588784 | 0.076337 | 62.055 |
| norway3g | 41 | 123 | -10.779114 | 6.790639 | 2.980815 | 145.440017 | 0.164449 | 60.002 |
| oboe | 100 | 300 | -12.779248 | 23.630938 | 3.147562 | 169.020214 | 0.127094 | 60.606 |
| hsr | 34 | 102 | -0.166975 | 6.954083 | 3.756841 | 35.863762 | 0.571334 | 62.174 |

This report does not certify independence of the Merina656 training pool from the test traces.
