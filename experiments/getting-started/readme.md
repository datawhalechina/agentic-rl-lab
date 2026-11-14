# 环境配置与基础概念

这一页是进入仓库的第一站：装好环境、注册两个账号、跑通第一次训练，顺带把读后续章节需要的几个概念讲清楚。对应 Datawhale 组队学习的 Task01。

全程不需要 GPU。训练和采样跑在 PyTRIO 的云端 GPU 上，你的电脑只负责数据、reward 和训练循环，一台能够联网的笔记本电脑就足够了。

## 0. 先搞清楚谁在干什么

| 组件 | 跑在哪 | 负责什么 |
| --- | --- | --- |
| PyTRIO（`pytrio==0.2.9`） | 云端 | 前向、反向、优化器更新、采样、权重保存 |
| SwanLab（`swanlab==0.10.4`） | 云端 + 本地日志 | 指标曲线、超参数、终端输出 |
| 本仓库的代码 | 本地 | 数据集、tokenizer、reward、训练循环，把上面两家串起来 |

本地依赖里的 torch 是 CPU 版（`pyproject.toml` 指定了 `pytorch-cpu` 源），不要装 CUDA 版。而且它只在写自定义损失函数时才用得上：`cross_entropy`、`importance_sampling`、`ppo` 这些内置 loss 全程在云端算，不碰本地 torch；走 `forward_backward_custom`（grpo 里的 cispo、DAPO、GSPO 等）时 loss 在本地跑 autograd，才需要它。仓库把 torch 放进基础依赖，是因为好几个章节用自定义 loss，且这些脚本在文件头就 import torch；自己写脚本只用内置 loss 的话，可以不装。

所以你要准备的东西只有三样：一个 Python >= 3.13 的环境（使用 uv 管理）、一个 PyTRIO 账号、一个 SwanLab 账号。

## 1. 装 uv，同步依赖

1. 安装 uv（装过就跳过）：

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows（PowerShell）
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   `brew install uv` 或 `pip install uv` 也可以。

2. clone 仓库并同步依赖：

   ```bash
   git clone https://github.com/datawhalechina/agentic-rl-lab.git
   cd agentic-rl-lab
   uv sync
   ```

   `uv sync` 按 `uv.lock` 创建 `.venv` 并装好全部依赖，版本和仓库完全一致。本地 Python 低于 3.13 时 uv 会自动下载一个合适的版本，不用手动装。

   **Intel 芯片 Mac 看这里**：`uv sync` 如果报下面这个错，不是网络问题，别反复重试——

   ```text
   error: Distribution `torch==2.13.0 @ registry+https://download.pytorch.org/whl/cpu` can't be installed
   because it doesn't have a source distribution or wheel for the current platform
   hint: You're on macOS (`macosx_15_0_x86_64`), but `torch` (v2.13.0) only has wheels for
   the following platform: `macosx_14_0_arm64`
   ```

   原因：PyTorch 从 2.3 起就不再为 macOS 的 x86_64（Intel）架构发布二进制轮子，只剩 Apple Silicon（arm64）的。本仓库要求 `torch>=2.9.1`，换镜像、降版本、加参数都绕不过去，Intel Mac 上原生装不上。Linux 的 x86_64 轮子是有的，所以走 Docker：

   ```bash
   cd agentic-rl-lab
   docker build -t agentic-rl-lab .     # 首次约 10-15 分钟（torch CPU 轮子约 200MB）
   docker run -it --name rl-lab agentic-rl-lab bash
   ```

   进容器后本章剩余步骤照常执行（`uv run trio login`、验证版本等），`uv sync` 已经在镜像构建时完成了。两点注意：容器里同样是 `uv run` 前缀；PyTRIO / SwanLab 的登录凭据存在容器内，用 `--name` 命名容器并 `docker start -ai rl-lab` 复用，别用 `--rm` 一次性容器，否则每次都要重新登录。临时跑单个脚本可以用：

   ```bash
   docker run --rm -v "$PWD:/app" -w /app agentic-rl-lab uv run python experiments/xxx.py
   ```

3. 记住一件事：仓库里的命令都加 `uv run` 前缀，或者先 `source .venv/bin/activate`（Windows 是 `.venv\Scripts\activate`）。`trio`、`swanlab` 这两个可执行文件在 `.venv` 里，在系统终端直接敲会报 command not found。

4. 验证版本：

   ```bash
   uv run python -c "from importlib.metadata import version; print(version('pytrio'), version('swanlab'))"
   ```

   输出应该是 `0.2.9 0.10.4`，大于这个版本的都可以～

5. 额外依赖按需装：跑 ALFWorld、TEMPO、AgentOPSD 三章需要 `uv sync --extra alfworld`，其余章节不用。

网络不顺畅时（顺畅就跳过）：

- `uv sync` 慢：`UV_DEFAULT_INDEX=https://mirrors.cernet.edu.cn/pypi/web/simple uv sync`。
- GSM8K 下载超时：`export HF_ENDPOINT=https://hf-mirror.com` 后重跑。部分章节的数据走 ModelScope，不需要翻墙，以各章 `start.md` 为准。

## 2. PyTRIO：注册、登录、验证

PyTRIO 是远程后训练服务：本地脚本把数据和 loss 发过去，云端算梯度、更新权重、返回采样结果，思路接近 Tinker 的 Train API。仓库里所有 RL 算法都跑在它上面。

注册和拿 key：

1. 打开官网 [pytrio.com](https://pytrio.com) 用邮箱注册，免费。`pytrio.cn` 会跳转到这里，是同一个服务。
2. 登录后进控制台 [pytrio.com/dashboard](https://pytrio.com/dashboard)，复制你的 API Key。

登录：

```bash
uv run trio login
```

终端会打印 key 的获取地址和粘贴提示。粘贴 key 时屏幕上什么都不显示，这是正常的，不是卡了，粘完直接回车。成功时长这样：

```text
trio: Login successfully. Hi, <your username>!
```

凭据保存在本地，这台机器之后不用再登录。注意命令是 `trio`，不是 `pytrio`。

CI 或脚本等不方便交互的场景，两种替代：

```bash
uv run trio login -k <your API Key>
# 或者
export PYTRIO_API_KEY=<your API Key>
```

部分章节的评测脚本（如 medical-opd）从自己目录下的 `.env` 读 `PYTRIO_API_KEY`；Search-R1 的 DeepSeek 搜索后端还需要 `DEEPSEEK_API_KEY`，Tournament-GRPO 需要 judge 模型的 key。这些都在各章 `start.md` 里写明，`.env` 已被 gitignore，别把 key 提交进仓库。

验证连接（几乎不消耗额度）：

```bash
uv run python - <<'PY'
import pytrio as trio
client = trio.ServiceClient()
for name in client.get_supported_models():
    print(name)
PY
```

能打印出 `Qwen/Qwen3.5-4B` 一类的模型清单就说明通了，清单以服务端实际提供为准。

关于额度：训练和采样消耗 PyTRIO 账户额度，余额和用量在控制台看。本仓库的 demo 都是小参数（几道题、几个采样、十几步），目的是做算法级复现。

卡住了先看官方文档 [docs.pytrio.com](https://docs.pytrio.com)，从快速开始读起。

## 3. SwanLab：注册、登录、选模式

SwanLab 负责实验记录：reward 和 loss 曲线、超参数、终端输出都进它。章节之间对比 loss、组队学习交训练记录，靠的都是它。国内服务，不用翻墙。

1. 在 [swanlab.cn](https://swanlab.cn) 注册账号。
2. 登录后到个人设置页复制 API Key：<https://swanlab.cn/space/~/settings#development>（登录命令也会打印这个地址）。
3. 登录并检查：

   ```bash
   uv run swanlab login
   uv run swanlab verify
   ```

   粘贴 key 时同样不回显。上传失败或模式意外回退时，先 `uv run swanlab ping` 查网络。

`swanlab.init(mode=...)` 有四种模式：

| 模式 | 数据去向 | 什么时候用 |
| --- | --- | --- |
| `online` | 本地 + 云端 | 默认，需要登录 |
| `local` | 仅本地，用 `swanlab watch` 看 | 不想注册账号 |
| `offline` | 仅本地，之后可 `swanlab sync` 补传 | 机器暂时没网 |
| `disabled` | 不记录 | 纯跑训练，不看曲线 |

各训练脚本的 `--swanlab-mode` 可选值略有差别：多数章节四种都有，grpo 的三个脚本只有 `online` 和 `disabled`，以 `--help` 为准。默认都是 `online`。

组队学习建议直接用 `online`：作业评审要训练记录链接，离线日志没法直接交。

```bash
uv run swanlab watch experiments/grpo/swanlog
```

浏览器打开 `http://127.0.0.1:5092`。

两个已知问题，先说了省得踩：

1. **watch 报 `Please install the swanboard package`**：dashboard 不在 `swanlab` 的默认依赖里，需要时按需补装（预发布版本，得开 prerelease）：

   ```bash
   uv pip install --prerelease=allow "swanlab[dashboard]==0.10.4"
   ```

   装进 `.venv` 即可，`uv sync` 不会动它；重建环境后要再装一次。

2. **Docker 容器里 watch 地址打不开**：watch 默认绑 `127.0.0.1`，那是容器自己的回环地址，宿主机浏览器到不了。起服务时改成绑 `0.0.0.0`：

   ```bash
   uv run swanlab watch -h 0.0.0.0 experiments/grpo/swanlog
   ```

   然后浏览器访问容器 IP。IP 用这个命令查：

   ```bash
   docker inspect <容器名> --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
   ```

   打开 `http://<容器IP>:5092`。注意容器重启后 IP 会变。想要固定用 `http://127.0.0.1:5092` 的话，建容器时就带上端口映射：`docker run -it --name rl-lab -p 5092:5092 agentic-rl-lab bash`，之后浏览器直接开 `http://127.0.0.1:5092`，两种方式都不受浏览器 DNS 设置影响（`<容器名>.orb.local` 这种域名写法部分浏览器解析不了，不推荐）。


## 4. 基础概念：读懂代码的最小词汇表

这里只讲到能读代码的程度，推导和对比在各章正文里。

**SFT 与 RL。** SFT 给标准答案，模型按 cross_entropy 学；RL 让模型自己作答，用 reward 打分，把好的回答强化、差的压下去。GRPO 往后的章节都是 RL 或蒸馏，SFT 只在 Spec-o3、Medical OPD 等章节作为热启动出现。

**policy、rollout、on-policy。** policy 就是当前正在训练的权重；从 prompt 出发生成一次回答叫一条 rollout。on-policy 指训练数据由当前权重采样而来，所以 demo 每个 step 先存权重拿 sampler、采完再更新。用旧权重采的数据是 off-policy，两者之间的偏差靠下面的重要性比率补偿。

**reward。** 给一条回答打的分。GSM8K 是 0/1（boxed 答案是否匹配）；Search-R1 会加上格式和搜索行为；ALFWorld 看任务是否完成。reward 设计往往比算法本身更影响结果，各章 readme 都会单独解释自己的 reward。

**advantage 与组。** GRPO 对同一道题采 `group_size` 个回答，advantage = reward - 组内平均 reward：比平均好的被鼓励，比平均差的被压低。不需要额外训一个 critic 模型，这是 GRPO 相对 PPO 最大的简化。整组 reward 全相同（全对或全错）时 advantage 全 0，这一组就白采了，也就是日志里的 degenerate group。

**old logprobs 与重要性比率。** logprob 是每个 token 的对数概率。同一 token 在新旧 policy 下的概率之比就是重要性采样比率，RL loss 基本都在对这个比率乘 advantage 做文章：`ppo` 给比率加 clip，`cispo` 换了 clip 的位置。直觉解释在 [Loss Functions](../loss-functions/readme.md) 一章，建议读 GRPO 之前先读它。

**LoRA 与权重。** 不动 base model，只训一个低秩适配器（demo 里 rank 32）。训练结束保存的就是这份适配器：`save_weights_for_sampler` 存给采样和评测用，`save_state` 存完整 checkpoint 用于续训。权重地址是 `trio://` 形式。

**OPD（On-Policy Distillation）。** 同样用 student 自己采样的回答，但信号不是 reward：teacher 模型对 student 的回答逐 token 算 logprobs，用 student 与 teacher 的差（reverse KL）当 advantage，相当于老师逐 token 批改。OPSD 再进一步，teacher 就是 step 0 的自己，只是 prompt 里多给了参考解答。见 [General OPD](../general-opd/readme.md)、[OPSD](../opsd/readme.md) 两章。

**多轮与 observation mask。** Agentic RL 让模型在回答中途调工具：搜索、代码解释器、TextWorld 环境。工具返回的内容（observation）拼进上下文给模型看，但不参与 loss——环境说的话不是模型的动作，不该被强化。代码里体现为这些 token 的 logprobs 和 advantages 填 0。见 [Search-R1](../search-r1/readme.md)、[ReTool](../retool/readme.md)、[ALFWorld](../alfworld/readme.md) 三章。


## 附：命令速查

```bash
uv sync                                          # 装 / 恢复依赖
uv run trio login                                # PyTRIO 登录
uv run swanlab login                             # SwanLab 登录
uv run swanlab verify                            # 检查 SwanLab 凭据
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --swanlab-mode online        # 第一次训练
uv run swanlab watch experiments/grpo/swanlog     # 看本地日志

# Intel Mac 走 Docker（见第 1 节）
docker build -t agentic-rl-lab .                  # 首次构建镜像
docker start -ai rl-lab                           # 复用已命名的容器
```
