# Environment Setup and Core Concepts

This page is the first stop in the repository: set up the environment, register two accounts, run your first training, and pick up the concepts needed for the later chapters along the way. It corresponds to Task01 of the Datawhale group study.

No GPU is needed at any point. Training and sampling run on PyTRIO's cloud GPUs; your own machine only handles data, rewards and the training loop, so a laptop with internet access is enough.

## 0. Who runs what

| Component | Where it runs | What it does |
| --- | --- | --- |
| PyTRIO (`pytrio==0.2.9`) | Cloud | Forward, backward, optimizer updates, sampling, weight storage |
| SwanLab (`swanlab==0.10.4`) | Cloud + local logs | Metric curves, hyperparameters, terminal output |
| This repository's code | Local | Datasets, tokenizer, rewards, training loop; wires the two services together |

The torch in the local dependencies is the CPU build (`pyproject.toml` pins the `pytorch-cpu` index); do not install a CUDA build. It is only needed when you write custom loss functions: built-in losses such as `cross_entropy`, `importance_sampling` and `ppo` are computed entirely in the cloud and never touch local torch. Only `forward_backward_custom` (cispo in grpo, DAPO, GSPO and others) runs the loss locally with autograd, and that is where torch is used. The repository keeps torch in the base dependencies because several chapters use custom losses and those scripts import torch at the top of the file; if you write your own scripts with built-in losses only, you can skip it.

So there are only three things to prepare: a Python >= 3.13 environment (managed by uv), a PyTRIO account, and a SwanLab account.

## 1. Install uv and sync dependencies

1. Install uv (skip if already installed):

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows（PowerShell）
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   `brew install uv` or `pip install uv` also work.

2. Clone the repository and sync dependencies:

   ```bash
   git clone https://github.com/datawhalechina/agentic-rl-lab.git
   cd agentic-rl-lab
   uv sync
   ```

   `uv sync` creates `.venv` from `uv.lock` and installs every dependency at exactly the repository's versions. If your local Python is older than 3.13, uv downloads a suitable version automatically; no manual install is needed.

3. Remember one thing: prefix every command in this repository with `uv run`, or activate first with `source .venv/bin/activate` (`.venv\Scripts\activate` on Windows). The `trio` and `swanlab` executables live inside `.venv`; typing them in a plain system terminal gives command not found.

4. Verify the versions:

   ```bash
   uv run python -c "from importlib.metadata import version; print(version('pytrio'), version('swanlab'))"
   ```

   The output should be `0.2.9 0.10.4`; newer versions are fine too.

5. Install extras only when needed: the ALFWorld, TEMPO and AgentOPSD chapters require `uv sync --extra alfworld`; the other chapters do not.

If your network is slow (skip this when it is not):

- Slow `uv sync`: `UV_DEFAULT_INDEX=https://mirrors.cernet.edu.cn/pypi/web/simple uv sync`.
- GSM8K download timeouts: `export HF_ENDPOINT=https://hf-mirror.com` and rerun. Some chapters fetch data from ModelScope, which needs no proxy; follow each chapter's `start.md`.

## 2. PyTRIO: register, log in, verify

PyTRIO is a remote post-training service: your local script sends data and losses, the cloud computes gradients, updates weights and returns sampling results — an approach close to Tinker's Train API. Every RL algorithm in this repository runs on it.

Register and get a key:

1. Sign up with email at [pytrio.com](https://pytrio.com); it is free. `pytrio.cn` redirects here and is the same service.
2. After logging in, open the console at [pytrio.com/dashboard](https://pytrio.com/dashboard) and copy your API key.

Log in:

```bash
uv run trio login
```

The terminal prints where to find the key and a paste prompt. Nothing appears on screen while you paste the key; that is normal, not a freeze — paste and press Enter. Success looks like this:

```text
trio: Login successfully. Hi, <your username>!
```

Credentials are stored locally, so this machine does not need to log in again. Note that the command is `trio`, not `pytrio`.

For CI or scripts where interactive login is inconvenient, two alternatives:

```bash
uv run trio login -k <your API Key>
# 或者
export PYTRIO_API_KEY=<your API Key>
```

Some chapters' evaluation scripts (medical-opd for example) read `PYTRIO_API_KEY` from a `.env` in their own directory; Search-R1's DeepSeek search backend also needs `DEEPSEEK_API_KEY`, and Tournament-GRPO needs a key for its judge model. Each chapter's `start.md` spells these out. `.env` is gitignored — do not commit keys.

Verify the connection (costs almost no quota):

```bash
uv run python - <<'PY'
import pytrio as trio
client = trio.ServiceClient()
for name in client.get_supported_models():
    print(name)
PY
```

If it prints a model list such as `Qwen/Qwen3.5-4B`, the connection works; the actual list depends on what the service currently offers.

About quota: training and sampling consume your PyTRIO account quota; the console shows balance and usage. The demos in this repository use small settings (a few questions, a few samples, a dozen steps) and aim at algorithm-level reproduction.

If you get stuck, read the official documentation at [docs.pytrio.com](https://docs.pytrio.com), starting from the quick start.

## 3. SwanLab: register, log in, pick a mode

SwanLab handles experiment tracking: reward and loss curves, hyperparameters and terminal output all go there. Comparing losses across chapters and submitting training records for the group study both rely on it. It is a China-based service and needs no proxy.

1. Register at [swanlab.cn](https://swanlab.cn).
2. After logging in, copy the API key from your settings page: <https://swanlab.cn/space/~/settings#development> (the login command prints this address too).
3. Log in and check:

   ```bash
   uv run swanlab login
   uv run swanlab verify
   ```

   Pasting the key is hidden here as well. If uploads fail or the mode falls back unexpectedly, run `uv run swanlab ping` first to check the network.

`swanlab.init(mode=...)` has four modes:

| Mode | Where data goes | When to use it |
| --- | --- | --- |
| `online` | Local + cloud | Default; requires login |
| `local` | Local only, viewed with `swanlab watch` | No account needed |
| `offline` | Local only, upload later with `swanlab sync` | Machine temporarily offline |
| `disabled` | Nothing recorded | Pure training runs without curves |

The allowed values of `--swanlab-mode` differ slightly per training script: most chapters offer all four, while the three grpo scripts only offer `online` and `disabled`; check `--help`. The default is `online` everywhere.

For the group study, use `online` directly: assignment review asks for a training-record link, and offline logs cannot be submitted as-is.

```bash
uv run swanlab watch experiments/grpo/swanlog
```

Then open `http://127.0.0.1:5092` in a browser.

## 4. Core concepts: the minimum vocabulary for reading the code

This section goes only as far as reading the code; derivations and comparisons live in the chapter bodies.

**SFT vs RL.** SFT supplies reference answers and the model learns with cross_entropy; RL lets the model answer on its own, scores the answers with rewards, and reinforces the good ones while suppressing the bad ones. Every chapter from GRPO onward is RL or distillation; SFT appears only as a warm start in chapters such as Spec-o3 and Medical OPD.

**Policy, rollout, on-policy.** The policy is the weights currently being trained; generating one answer from a prompt is a rollout. On-policy means the training data is sampled from the current weights, which is why the demo saves weights to a sampler at each step, samples, and only then updates. Data sampled from older weights is off-policy, and the gap between the two is compensated by the importance ratio below.

**Reward.** The score given to one answer. GSM8K uses 0/1 (whether the boxed answer matches); Search-R1 adds format and search behaviour; ALFWorld checks task completion. Reward design often matters more than the algorithm itself, and each chapter's readme explains its own reward.

**Advantage and the group.** GRPO samples `group_size` answers per question and sets advantage = reward - mean reward of the group: answers above the mean are encouraged, answers below it are suppressed. No extra critic model is trained, which is GRPO's biggest simplification over PPO. When every reward in a group is identical (all correct or all wrong), all advantages are 0 and the group contributes nothing — the degenerate group in the logs.

**Old logprobs and the importance ratio.** A logprob is the log probability of a token. The ratio of a token's probability under the new and old policies is the importance sampling ratio, and RL losses mostly work on this ratio times the advantage: `ppo` clips the ratio, `cispo` moves where the clip applies. The intuition is in the [Loss Functions](../loss-functions/readme.md) chapter; read it before GRPO.

**LoRA and weights.** The base model stays frozen; only a low-rank adapter is trained (rank 32 in the demo). What gets saved after training is this adapter: `save_weights_for_sampler` stores it for sampling and evaluation, `save_state` stores a full checkpoint for resuming. Weight addresses use the `trio://` form.

**OPD (On-Policy Distillation).** Still uses the student's own sampled answers, but the signal is not a reward: a teacher model computes logprobs over the student's answers token by token, and the student-teacher gap (reverse KL) becomes the advantage — like a teacher grading token by token. OPSD goes one step further: the teacher is the step-0 self, except its prompt also contains the reference solution. See the [General OPD](../general-opd/readme.md) and [OPSD](../opsd/readme.md) chapters.

**Multi-turn and observation masks.** Agentic RL lets the model call tools mid-answer: search, a code interpreter, TextWorld environments. Tool returns (observations) are appended to the context for the model to read but do not enter the loss — words from the environment are not the model's actions and should not be reinforced. In code, these tokens get 0 for logprobs and advantages. See the [Search-R1](../search-r1/readme.md), [ReTool](../retool/readme.md) and [ALFWorld](../alfworld/readme.md) chapters.

## Appendix: command cheat sheet

```bash
uv sync                                          # 装 / 恢复依赖
uv run trio login                                # PyTRIO 登录
uv run swanlab login                             # SwanLab 登录
uv run swanlab verify                            # 检查 SwanLab 凭据
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --swanlab-mode online        # 第一次训练
uv run swanlab watch experiments/grpo/swanlog     # 看本地日志
```