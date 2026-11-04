# Qwen ROI pilot

逐帧候选证据定位；无答案输入。时间约束需在后续聚合阶段处理。

| Frame | t (s) | ROI count | Pixel boxes | Overlay |
|---|---:|---:|---|---|
| dev10_01_f00 | 0.0 | 0 | [] | [image](overlays/dev10_01_f00.jpg) |
| dev10_01_f01 | 1.0 | 0 | [] | [image](overlays/dev10_01_f01.jpg) |
| dev10_01_f02 | 2.0 | 0 | [] | [image](overlays/dev10_01_f02.jpg) |
| dev10_01_f03 | 3.0 | 1 | [[864, 406, 1042, 547]] | [image](overlays/dev10_01_f03.jpg) |
| dev10_01_f04 | 4.0 | 1 | [[866, 396, 1046, 535]] | [image](overlays/dev10_01_f04.jpg) |
| dev10_01_f05 | 4.8 | 1 | [[855, 404, 1023, 546]] | [image](overlays/dev10_01_f05.jpg) |
| dev10_02_f00 | 0.0 | 1 | [[1562, 384, 1646, 445]] | [image](overlays/dev10_02_f00.jpg) |
| dev10_02_f01 | 1.0 | 0 | [] | [image](overlays/dev10_02_f01.jpg) |
| dev10_02_f02 | 2.0 | 0 | [] | [image](overlays/dev10_02_f02.jpg) |
| dev10_02_f03 | 3.0 | 1 | [[1678, 619, 1778, 653]] | [image](overlays/dev10_02_f03.jpg) |
| dev10_02_f04 | 4.0 | 0 | [] | [image](overlays/dev10_02_f04.jpg) |
| dev10_02_f05 | 4.8 | 0 | [] | [image](overlays/dev10_02_f05.jpg) |
| dev10_03_f00 | 0.0 | 1 | [[1136, 645, 1224, 700]] | [image](overlays/dev10_03_f00.jpg) |
| dev10_03_f01 | 1.0 | 1 | [[1743, 903, 1836, 966]] | [image](overlays/dev10_03_f01.jpg) |
| dev10_03_f02 | 2.0 | 0 | [] | [image](overlays/dev10_03_f02.jpg) |
| dev10_03_f03 | 3.0 | 0 | [] | [image](overlays/dev10_03_f03.jpg) |
| dev10_03_f04 | 4.0 | 1 | [[1228, 729, 1310, 778]] | [image](overlays/dev10_03_f04.jpg) |
| dev10_03_f05 | 4.8 | 1 | [[1186, 727, 1275, 788]] | [image](overlays/dev10_03_f05.jpg) |
