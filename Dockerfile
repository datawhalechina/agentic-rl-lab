# 供 Intel Mac (x86_64) 运行 agentic-rl-lab 用：
# PyTorch 官方已停止发布 macOS x86_64 轮子（arm64-only），
# 但 Linux x86_64 有 torch 2.13.0+cpu 轮子，故用 linux/amd64 容器承接。
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# 加速镜像内文件写入，避免挂载时的硬链接报错
ENV UV_LINK_MODE=copy

# 先只拷贝依赖描述文件，充分利用 Docker 层缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 再拷贝项目代码，完成完整安装
COPY . .
RUN uv sync --frozen

# 交互式运行示例：
#   docker build -t agentic-rl-lab .
#   docker run --rm -it agentic-rl-lab
# 运行实验脚本示例：
#   docker run --rm -v "$PWD:/app" agentic-rl-lab uv run python experiments/xxx.py
CMD ["bash"]
