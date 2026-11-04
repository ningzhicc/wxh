# 碗：API框、官方掩码与编码保护区逐帧审查

本次检查原生分辨率的6帧，不调用API，不修改定位结果或GT。绿色为官方掩码，红色为API框，橙色为实际编码保护矩形。

结论：已检查帧的掩码基本贴着可见碗身；8秒框取局部、9秒错到手臂、12秒偏左，API空间误差是低覆盖的直接原因。掩码总帧数与视频帧数不同的全局疑点仍保留，不能据此宣称全部帧对齐；本次没有选择偏移抬高指标。

| 时间 | API覆盖GT像素 | 编码保护矩形覆盖 | 判断 |
|---|---:|---:|---|
| 8.000s | 53.84% | 91.65% | API框位于碗内但只取下部局部；掩码基本贴合可见碗身。 |
| 9.000s | 0.00% | 24.27% | API框落在手腕/手臂区域，实际碗在右下方部分被遮挡；掩码覆盖其可见部分。 |
| 9.833s | 69.95% | 98.01% | API命中碗，框遗漏下部边缘；官方掩码基本贴合可见轮廓。 |
| 10.000s | 89.02% | 100.00% | API命中碗且大部分重合，仍有边缘未覆盖。 |
| 11.000s | 93.91% | 100.00% | API框与碗主体及掩码基本吻合。 |
| 12.000s | 45.47% | 84.24% | API框相对实际碗明显向左偏，漏掉右侧碗身。 |

API均值58.7%、编码矩形83.0%维持原值，80%的预设API门槛仍未通过。编码静态并集和padding扩大了保护范围，不能把它的覆盖率当成API准确率。

对颜色问题，局部有效颜色证据可能足以回答，但这不等于整物体定位合格；后续同时报告目标命中、空间覆盖、编码保护覆盖和接收端任务正确率。
本次6帧目视判断由助手完成，不冒充人工GT或完整跟踪标注。

## 8.000s

![原图与几何对比](/home/amora/wxh/jev/semantic_streaming/artifacts/bowl_geometry_review_v1/chunk_0001_frame000048.png)

## 9.000s

![原图与几何对比](/home/amora/wxh/jev/semantic_streaming/artifacts/bowl_geometry_review_v1/chunk_0001_frame000054.png)

## 9.833s

![原图与几何对比](/home/amora/wxh/jev/semantic_streaming/artifacts/bowl_geometry_review_v1/chunk_0001_frame000059.png)

## 10.000s

![原图与几何对比](/home/amora/wxh/jev/semantic_streaming/artifacts/bowl_geometry_review_v1/chunk_0002_frame000060.png)

## 11.000s

![原图与几何对比](/home/amora/wxh/jev/semantic_streaming/artifacts/bowl_geometry_review_v1/chunk_0002_frame000066.png)

## 12.000s

![原图与几何对比](/home/amora/wxh/jev/semantic_streaming/artifacts/bowl_geometry_review_v1/chunk_0002_frame000072.png)

