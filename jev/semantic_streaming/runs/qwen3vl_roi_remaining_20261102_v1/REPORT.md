# Qwen ROI pilot

逐帧候选证据定位；无答案输入。时间约束需在后续聚合阶段处理。

| Frame | t (s) | ROI count | Pixel boxes | Overlay |
|---|---:|---:|---|---|
| dev10_04_f00 | 0.0 | 1 | [[470, 252, 793, 300]] | [image](overlays/dev10_04_f00.jpg) |
| dev10_04_f01 | 1.0 | 1 | [[518, 430, 740, 477]] | [image](overlays/dev10_04_f01.jpg) |
| dev10_04_f02 | 2.0 | 1 | [[518, 430, 740, 477]] | [image](overlays/dev10_04_f02.jpg) |
| dev10_04_f03 | 3.0 | 1 | [[516, 429, 740, 478]] | [image](overlays/dev10_04_f03.jpg) |
| dev10_04_f04 | 4.0 | 1 | [[516, 430, 740, 482]] | [image](overlays/dev10_04_f04.jpg) |
| dev10_04_f05 | 4.8 | 1 | [[721, 633, 941, 761]] | [image](overlays/dev10_04_f05.jpg) |
| dev10_05_f00 | 0.0 | 0 | [] | [image](overlays/dev10_05_f00.jpg) |
| dev10_05_f01 | 1.0 | 0 | [] | [image](overlays/dev10_05_f01.jpg) |
| dev10_05_f02 | 2.0 | 0 | [] | [image](overlays/dev10_05_f02.jpg) |
| dev10_05_f03 | 3.0 | 0 | [] | [image](overlays/dev10_05_f03.jpg) |
| dev10_05_f04 | 4.0 | 1 | [[200, 262, 305, 403]] | [image](overlays/dev10_05_f04.jpg) |
| dev10_05_f05 | 4.8 | 1 | [[200, 262, 305, 403]] | [image](overlays/dev10_05_f05.jpg) |
| dev10_06_f00 | 0.0 | 1 | [[912, 414, 957, 467]] | [image](overlays/dev10_06_f00.jpg) |
| dev10_06_f01 | 1.0 | 1 | [[823, 610, 905, 850]] | [image](overlays/dev10_06_f01.jpg) |
| dev10_06_f02 | 2.0 | 1 | [[1626, 953, 1700, 1005]] | [image](overlays/dev10_06_f02.jpg) |
| dev10_06_f03 | 3.0 | 1 | [[514, 609, 590, 674]] | [image](overlays/dev10_06_f03.jpg) |
| dev10_06_f04 | 4.0 | 1 | [[485, 479, 538, 534]] | [image](overlays/dev10_06_f04.jpg) |
| dev10_06_f05 | 4.8 | 1 | [[347, 385, 749, 856]] | [image](overlays/dev10_06_f05.jpg) |
| dev10_07_f00 | 0.0 | 1 | [[1553, 862, 1846, 1009]] | [image](overlays/dev10_07_f00.jpg) |
| dev10_07_f01 | 1.0 | 1 | [[1553, 864, 1846, 1009]] | [image](overlays/dev10_07_f01.jpg) |
| dev10_07_f02 | 2.0 | 1 | [[1553, 864, 1848, 1009]] | [image](overlays/dev10_07_f02.jpg) |
| dev10_07_f03 | 3.0 | 1 | [[1553, 864, 1846, 1009]] | [image](overlays/dev10_07_f03.jpg) |
| dev10_07_f04 | 4.0 | 1 | [[1553, 864, 1846, 1008]] | [image](overlays/dev10_07_f04.jpg) |
| dev10_07_f05 | 4.8 | 1 | [[1553, 864, 1846, 1009]] | [image](overlays/dev10_07_f05.jpg) |
| dev10_08_f00 | 0.0 | 1 | [[919, 927, 1039, 993]] | [image](overlays/dev10_08_f00.jpg) |
| dev10_08_f01 | 1.0 | 1 | [[919, 927, 1039, 993]] | [image](overlays/dev10_08_f01.jpg) |
| dev10_08_f02 | 2.0 | 1 | [[919, 927, 1039, 993]] | [image](overlays/dev10_08_f02.jpg) |
| dev10_08_f03 | 3.0 | 1 | [[919, 927, 1039, 993]] | [image](overlays/dev10_08_f03.jpg) |
| dev10_08_f04 | 4.0 | 1 | [[919, 927, 1039, 993]] | [image](overlays/dev10_08_f04.jpg) |
| dev10_08_f05 | 4.8 | 1 | [[919, 928, 1039, 993]] | [image](overlays/dev10_08_f05.jpg) |
| dev10_09_f00 | 0.0 | 1 | [[984, 444, 1053, 472]] | [image](overlays/dev10_09_f00.jpg) |
| dev10_09_f01 | 1.0 | 1 | [[1021, 339, 1129, 385]] | [image](overlays/dev10_09_f01.jpg) |
| dev10_09_f02 | 2.0 | 0 | [] | [image](overlays/dev10_09_f02.jpg) |
| dev10_09_f03 | 3.0 | 0 | [] | [image](overlays/dev10_09_f03.jpg) |
| dev10_09_f04 | 4.0 | 1 | [[960, 525, 1010, 548]] | [image](overlays/dev10_09_f04.jpg) |
| dev10_09_f05 | 4.8 | 0 | [] | [image](overlays/dev10_09_f05.jpg) |
| dev10_10_f00 | 0.0 | 1 | [[1088, 775, 1208, 835]] | [image](overlays/dev10_10_f00.jpg) |
| dev10_10_f01 | 1.0 | 1 | [[1115, 773, 1235, 835]] | [image](overlays/dev10_10_f01.jpg) |
| dev10_10_f02 | 2.0 | 0 | [] | [image](overlays/dev10_10_f02.jpg) |
| dev10_10_f03 | 3.0 | 0 | [] | [image](overlays/dev10_10_f03.jpg) |
| dev10_10_f04 | 4.0 | 0 | [] | [image](overlays/dev10_10_f04.jpg) |
| dev10_10_f05 | 4.8 | 0 | [] | [image](overlays/dev10_10_f05.jpg) |
