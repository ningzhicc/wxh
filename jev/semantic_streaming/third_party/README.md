# 外部材料与本地工具

- 问答与元数据：`pku-netvideo/DeViBench`，固定 commit 见数据集 `sources.lock.json`，原始 MIT 文件保存在来源快照目录。
- 视频：`mjuicem/StreamingBench`，固定 HF revision 见同一 lock；所查 dataset card 没有声明统一的视频许可证，原始上传链接也不在所查 CSV 中。未将 MIT 当作所有视频的许可。
- FFmpeg 启动工具：从 `imageio-ffmpeg==0.6.0` 的 Linux x86_64 wheel 提取，包及二进制 SHA256 见 `configs/tools.lock.json`。imageio-ffmpeg 包的 BSD 许可见本目录 `imageio_ffmpeg/LICENSE`。
- 捆绑 FFmpeg 是 John Van Sickle 的 7.0.2 静态构建，`ffmpeg -version` 明示启用 GPL/version3；不要把 Python 包 BSD 许可视为捆绑二进制的全部许可。来源与构建信息：https://johnvansickle.com/ffmpeg/ ，https://github.com/imageio/imageio-ffmpeg 。二进制仅保留本地且被 Git 忽略。
