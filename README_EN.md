<div align="center">

<a href="https://github.com/datawhalechina/agentic-rl-lab">
  <img src="images/agentic-rl-lab.png" alt="Agentic-RL Lab" width=100% />
</a>

<br>

<p>
  <a href="./README.md">中文</a> | <strong>English</strong>
</p>

<p>
  Reproduce and dissect frontier LLM reinforcement learning algorithms — GRPO, OPD, OPSD, GSPO, DAPO, Search-R1, ReTool, ALFWorld, Vision GRPO, TEMPO, AgentOPSD, Spec-o3, PyTrio-Jev, Slime and more — with simpler code and a lower GPU barrier.
</p>

<p>
  <a href="https://pytrio.cn/"><img alt="PyTRIO" src="https://img.shields.io/badge/PyTRIO-Remote%20Training-d94a45?style=flat" /></a>
  <a href="https://swanlab.cn/"><img alt="SwanLab" src="https://img.shields.io/badge/SwanLab-Experiment%20Tracking-258f4b?style=flat" /></a>
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6?style=flat" /></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49?style=flat" /></a>
  <a href="https://github.com/datawhalechina/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab&amp;label=visitors&amp;color=1283c3&amp;style=flat" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13%2B-306998?style=flat" />
</p>

</div>

<p align="center">
  💬 <strong>WeChat Group:</strong>
  <a href="https://github.com/datawhalechina/agentic-rl-lab/issues/13">Click to view the group QR code</a>
</p>

<p align="center">
  📖 <strong>Read online:</strong>
  <a href="https://datawhalechina.github.io/agentic-rl-lab/en/">English handbook</a> ·
  <a href="https://datawhalechina.github.io/agentic-rl-lab/">简体中文</a>
</p>

## Introduction

This is a lab notebook and tutorial-style repository. I use [PyTRIO](https://pytrio.com) to reproduce a set of reinforcement learning algorithms related to LLM / Agent RL, with three goals in mind:

1. Explain each algorithm clearly first: which paper it comes from, what problem it solves, and what its core variables are.
2. Then reproduce it with runnable code: data, reward, loss, training loop, and SwanLab tracking are all in this repo.
3. Maybe build a friendlier and more lightweight training framework for Agent RL and Harness RL in the future~

> *Why PyTRIO? I had long wanted to dig deep into Agentic-RL algorithms, but kept being held back by practical obstacles: no GPUs, the complexity of tightly coupled training-and-inference codebases, the highly coupled engineering of Verl, and so on. PyTRIO changed that — it lets me study Agentic-RL algorithms with much simpler code and a much lower barrier. I learned and reproduced everything in this repository in less than a month.*

> *I believe products like PyTRIO or Tinker are future-facing infrastructure for LLM post-training. Getting hands-on with them early is valuable for both algorithm engineers and researchers.*

If you have a promising Agentic-RL idea and want to turn it into a paper or a project, you are welcome to email us to apply for PyTRIO credits. The review process is fairly strict, so please write your proposal carefully and clearly in the email. PyTRIO is also happy to sponsor Agentic RL projects and papers — contact: violin@pytrio.com

## Table of Contents

| Topic | Content | Read online |
| --- | --- | --- |
| [Loss Function](./experiments/loss-functions/readme.md) | An intuitive explanation of what `importance_sampling`, `ppo`, and `cispo` are each optimizing | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/loss-functions/) |
| [GRPO](./experiments/grpo/readme.md) | Reproduce GRPO on GSM8K and compare the `importance_sampling` / `ppo` / `cispo` losses | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/grpo/) |
| [Vision GRPO](./experiments/vision-grpo/readme.md) | Add image inputs on GeoQA and train Qwen3.5-4B to solve geometry multiple-choice questions with rule-based rewards | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/vision-grpo/) |
| [PyTrio-Jev](./experiments/pytrio-jev/readme.md) | For the price of a McDonald's breakfast, anyone can train their own Jev decision model | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/pytrio-jev/) |
| [General OPD](./experiments/general-opd/readme.md) | A minimal closed loop on DeepMath-103K: Student sampling, Teacher scoring, and reverse KL | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/general-opd/) |
| [Medical OPD](./experiments/medical-opd/readme.md) | Starting from Medical SFT, boost medical capability with SAR-OPD and IDT-OPD while preserving general ability | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/medical-opd/) |
| [DAPO](./experiments/dapo/readme.md) | Dissect the four core improvements and measure the real time cost of Dynamic Sampling in training | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/dapo/) |
| [GSPO](./experiments/gspo/readme.md) | Lift the importance ratio and clipping from the token level to the sequence level | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/gspo/) |
| [OPSD](./experiments/opsd/readme.md) | Distill the Student's self-sampled trajectories with a fixed step-0 Teacher | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/opsd/) |
| [AgentOPSD](./experiments/agentopsd/readme.md) | Use a skill-guided Self-Teacher for turn-level credit assignment | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/agentopsd/) |
| [Search-R1](./experiments/search-r1/readme.md) | Reproduce multi-turn search RL with Qwen3.5-4B, PyTRIO, and switchable online search backends | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/search-r1/) |
| [ReTool](./experiments/retool/readme.md) | Reproduce code-interleaved Agentic RL with Qwen3.5-4B, PyTRIO, and a local code sandbox | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/retool/) |
| [ALFWorld](./experiments/alfworld/readme.md) | Train a household agent with 12K-token long trajectories, a real TextWorld environment, and group-relative advantage | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/alfworld/) |
| [TEMPO](./experiments/tempo/readme.md) | Algorithm-level reproduction of macro-step optimization with a generative critic | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/tempo/) |
| [Spec-o3](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/spec-o3/readme.md) | Find stars with reinforcement learning: use SFT + GRPO to interleave spectral inspection, reasoning, and tool use | [Read](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/spec-o3/) |

## Quick Start

If you want to clone this repository directly:

```bash
git clone https://github.com/datawhalechina/agentic-rl-lab.git
cd agentic-rl-lab
uv sync
```

For Spec-o3 data preparation, SFT, GRPO, evaluation, and plotting commands, see the [Spec-o3 Quick Start](https://datawhalechina.github.io/agentic-rl-lab/en/experiments/spec-o3/run.html).

New Spec-o3 RL epoch 3 results on the same 256 development examples: Qwen3.5-4B answers **175 correctly (68.36%)**, and Qwen3.5-9B answers **191 correctly (74.61%)**, with Macro F1 scores of **70.93% / 75.94%**, respectively. See the [Spec-o3 article](./experiments/spec-o3/readme.md) for the full comparison and evaluation settings.

To run the ALFWorld, TEMPO, or AgentOPSD chapters, you also need the TextWorld environment dependencies:

```bash
uv sync --extra alfworld
```

If you only want to pull a single demo script into your own project, the current base dependencies are:

```bash
uv add \
  "datasets>=5.0.0" \
  "httpx>=0.28.1" \
  "math-verify>=0.9.0" \
  "matplotlib>=3.11.0" \
  "modelscope>=1.38.1" \
  "numpy>=2.5.1" \
  "openai>=2.44.0" \
  "python-dotenv>=1.2.2" \
  "pytrio==0.2.9" \
  "swanlab==0.9.2" \
  "torch>=2.9.1" \
  "torchvision>=0.27.1" \
  "tqdm>=4.68.3"
```

Search-R1 calls the official DeepSeek Search API through `httpx`. See the [quickstart](./experiments/search-r1/start.md) for API key setup.

The ALFWorld optional extra corresponds to:

```bash
uv add --optional alfworld "alfworld==0.4.2" "spacy==3.8.13"
```

## Star History

<div align="center">
  <img src="./images/star-history-2026826.png" alt="GitHub Star History" width="700" />
</div>

## Contributor

Want to contribute to this repo? Welcome! Please read the [Contributing Guide](./CONTRIBUTING_EN.md) first: open an Issue proposal before starting a new algorithm chapter; fixes and docs improvements can go straight to a PR.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/agentic-rl-lab/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/agentic-rl-lab" />
  </a>
</div>

## License

See [LICENSE](./LICENSE).
