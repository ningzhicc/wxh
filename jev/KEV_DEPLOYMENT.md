# Kev 本地部署

按照 `OPEN_SOURCE_JEV_CATALOG.md` 的首选项目部署 Kev-4B，源码来自
[jaredpalmer/kev](https://github.com/jaredpalmer/kev)。部署目录为 `/home/amora/wxh/jev`。

## 启动和停止

```bash
cd /home/amora/wxh/jev
python3 kev_service.py start
python3 kev_service.py status
python3 kev_service.py stop
```

`start` 在后台启动服务，模型加载期间 `status` 会提示接口尚未就绪。
日志在 `runtime/kev.log`，PID 在 `runtime/kev.pid`。
服务监听 `127.0.0.1:8009`，启动时只读取本地模型缓存；不设置开机自启。
API 文档地址为 <http://127.0.0.1:8009/docs>。

## 六档码率请求

```bash
cd /home/amora/wxh/jev
python3 check_abr.py
curl --noproxy '*' http://127.0.0.1:8009/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @abr_request.json
```

`abr_request.json` 提供示例状态和 300、750、1200、1850、2850、4300 kbps 六个候选。
调用时请替换为实际状态及码率表。候选键就是码率数值字符串，映射为
`bitrate_kbps = int(response['answers']['bitrate']['choice'])`。
`probabilities` 是所有六个候选的概率，`confidence` 是接口自己的置信度指标。
验证结果保存到 `runtime/abr_response.json`，包含服务端与客户端延迟。
这个示例验证接口和动作映射；ABR 的实际 QoE 和概率校准需在网络轨迹上另行评估。

## Python 接入

使用 `kev/.venv/bin/python` 运行客户端：

```python
import json
from typesafe_sdk import TypeSafeClient, Choice

with open('/home/amora/wxh/jev/abr_request.json', encoding='utf-8') as f:
    request = json.load(f)
q = request['questions']['bitrate']
with TypeSafeClient(api_key='local', base_url='http://127.0.0.1:8009',
                   model='kev-latest') as client:
    response = client.system_one(
        state=request['state'],
        questions={'bitrate': Choice(instructions=q['instructions'], criteria=q['criteria'])},
    )
    print(int(response.choices['bitrate'].choice))
    print(response.choices['bitrate'].probabilities)
```

## 环境及复现

- 源码：`kev/`；源码提交、模型 revision、基座 revision、温度和 adapter 校验值见 `deployment.json`。
- 模型：`jaredpalmer/kev-4b@v1.0`，服务启动时使用包含完整提交 SHA 的本地 snapshot 路径。
- 独立环境：`kev/.venv/`，Python 3.12，服务使用 CUDA 和 BF16。
- 模型缓存：`.cache/huggingface/`；下载工具：`.tools/`；依赖缓存：`.cache/uv/`。
- 上游源码没有部署用的本地修改；管理脚本和示例位于源码目录外。
- 部分 CUDA 依赖从本机已有安装复制，版本与 `uv.lock` 完全一致，复制前逐文件核对 wheel RECORD 校验值；来源记录在 `reused_dependencies.json`。复制后的文件独立保存在 `.venv`，不依赖原环境目录。
- 实际安装版本记录在 `requirements.installed.txt`。使用官方基础 PyTorch 推理路径，未额外安装 `flash-linear-attention` 和 `causal-conv1d` 加速内核。

重新安装锁定依赖：

```bash
cd /home/amora/wxh/jev/kev
UV_CACHE_DIR=/home/amora/wxh/jev/.cache/uv \
  ../.tools/bin/uv sync --extra serve --python /home/amora/anaconda3/bin/python3 --frozen
cd /home/amora/wxh/jev
kev/.venv/bin/python download_kev.py
```

官方接口测试（服务就绪后）：

```bash
cd /home/amora/wxh/jev/kev
NO_PROXY=127.0.0.1,localhost KEV_BASE_URL=http://127.0.0.1:8009 \
  .venv/bin/python -m pytest tests/test_api.py -q
```

## 本次验证结果

2026-10-13 在 RTX 5090 D 上完成验证：

- PyTorch 2.8.0 / CUDA 12.8，Transformers 5.17.0，BF16 GPU 推理。
- 官方 API 测试：`12 passed, 1 skipped in 1.72s`；跳过项只适用于启用截断的服务，本部署默认拒绝超长状态。
- 六档码率示例返回所有六个概率，概率之和为 0.9999（接口四舍五入），两次请求均选择 1850 kbps。
- 示例服务端延迟：新状态 289.8 ms，重复状态 41.6 ms；客户端分别为 299.95 ms 和 46.66 ms。这是两次示例请求的观测值，不是延迟分布评测。
- 原始结果：`runtime/verification.json`、`runtime/api_tests.log`、`runtime/abr_response.json`。
- 服务已经在后台运行；重启机器后需执行 `python3 kev_service.py start`。
