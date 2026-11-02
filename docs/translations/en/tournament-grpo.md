# Tournament-GRPO: Putting 8 Research Reports Through an Elimination Tournament

> Want to skip the article and run it directly? See the [quick start guide](./start.md): prepare data → configure tools and judges → train → evaluate.

<div align="center">
  <img src="./images/封面.png" alt="Tournament-GRPO: Putting 8 Research Reports Through an Elimination Tournament" width="100%">
</div>

<div align="center">
  <a href="https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/tournament-grpo"><img alt="Code" src="https://img.shields.io/badge/Code-tournament--grpo-2563eb?style=flat"></a>
  <a href="https://arxiv.org/abs/2605.26958v2"><img alt="Paper" src="https://img.shields.io/badge/Paper-Tournament--GRPO-d94a45?style=flat"></a>
  <a href="https://github.com/PuffYang/Tournament_GRPO/tree/eea826cf66705cac9ec81823f746d0ccd6074228"><img alt="Author Code" src="https://img.shields.io/badge/Author%20Code-eea826cf-f59e0b?style=flat"></a>
  <a href="https://docs.pytrio.com/docs"><img alt="PyTRIO" src="https://img.shields.io/badge/PyTRIO-0.2.9-7c3aed?style=flat"></a>
  <a href="https://swanlab.cn/@MRLE/tournament-grpo"><img alt="SwanLab" src="https://img.shields.io/badge/SwanLab-tournament--grpo-16a34a?style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - This chapter's implementation: [training](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/tournament-grpo/train.py) / [tournament rewards](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/tournament-grpo/reward.py) / [multi-turn rollouts](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/tournament-grpo/rollout.py) / [judge](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/tournament-grpo/judge.py)
> - Paper: [Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation(EMNLP'26)](https://arxiv.org/abs/2605.26958v2)
> - Authors' code: [PuffYang/Tournament_GRPO](https://github.com/PuffYang/Tournament_GRPO/tree/eea826cf66705cac9ec81823f746d0ccd6074228)
> - SwanLab: [20-step training records](https://swanlab.cn/@MRLE/tournament-grpo/v1/x1tb4s/runs/s474uc49/chart)
> - Search and judging: [DeepSeek Search API](https://api-docs.deepseek.com/zh-cn/guides/search/) / [Jina Reader](https://jina.ai/reader/) / [Jev](https://docs.typesafe.ai/api/system-one) `jev-1.14.0`
> - Chapter author: [MRLE](https://mrlepro.com)



The earlier [Search-R1](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/search-r1/readme.md) and [ReTool](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/retool/readme.md) chapters share a convenient premise: match the reference answer, and the reward is 1. This chapter switches to deep research: the model searches, reads webpages, and ultimately writes a research report. For example, one question in the training set asks:

> What progress has been made in the application of data mining in materials science research?

There is no standard answer to this kind of question. Two reports might both be correct, while one offers broader coverage and stronger evidence. The reward has to come from an LLM judge evaluating the report against the question's rubric.

Tournament-GRPO asks how a judge should assign scores so that GRPO receives a useful signal. Its answer is straightforward: instead of scoring each report independently, have reports for the same question compete in pairs and calculate rewards from their wins.

I ported the authors' verl implementation to PyTRIO: Qwen3.5-4B LoRA, DeepSeek for search, Jina for reading webpages, and Jev as the judge. We will start with a map, then go through the algorithm, code, and experiments.

## 0. Results First

| Item | Status |
| --- | --- |
| Tournament reward, format reward, advantage, PPO loss | Numerical values and gradients checked individually against the authors' functions |
| 20-step training (8 questions × 8 trajectories) | 1,280 trajectories, 4,480 judge comparisons, answer completion rate 64% → 97% |
| Base vs. checkpoint evaluation | All 100 Deep Research Bench questions, RACE Overall 20.65 → 34.70 |
| Cost and time | Training: 78.92 yuan (4 hours 3 minutes) + evaluation: 13.12 yuan, totaling 92.04 yuan |

After 20 steps, the most obvious change was that the model learned to hand in its report on time. Across the 100 evaluation questions, answer completion rose from 60% to 98%, and format compliance from 42% to 72%. Most of the 14-point RACE Overall gain came from fewer blank reports: an empty report scores 0, and Base failed to submit a report on 40 questions. On the 59 questions where both models produced reports, Overall was 34.49 → 35.25.

## 1. Start with a Map

The training loop for rubric-based RL looks roughly like this:

```math
\text{Prompt} \rightarrow \text{Rubric} \rightarrow \text{Judge} \rightarrow \text{Reward} \rightarrow \text{GRPO} \rightarrow \text{Policy}
```

For each question, first prepare a set of rubrics—the scoring criteria for that question. The policy samples multiple responses to the same question. The judge evaluates them using the question, responses, and rubrics, aggregating the scores into a reward for each response. GRPO computes advantages from the relative rewards among responses to the same question, updates the policy, and starts the next round of sampling.

Tournament-GRPO changes only the **Judge → Reward** segment of this map:

| Node | What Tournament-GRPO does |
| --- | --- |
| Rubric | Generated for each question with GPT-4o before training and kept unchanged during training (static rubrics) |
| Judge | Reads 2 responses and the complete rubric at a time, then selects the better response |
| Reward | Accumulates wins over multiple tournament rounds, applies min-max normalization within the group, then adds a 0/1 format reward |
| GRPO / Policy | Standard GRPO group-standardized advantages and PPO clipping, unchanged |

As you read this chapter, keep one question in mind: **With the same judge, how does changing the way we ask affect the signal that GRPO receives?**

## 2. Where Does Tournament-GRPO Come From, and What Does It Solve?

>Tournament-GRPO was proposed by Zixuan Yang et al. and has been **accepted to the EMNLP 2026 Main Conference**. The full paper title is *Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation*.

The paper groups existing approaches under pointwise scoring. They share one feature: the judge sees only one response at a time, without access to the other responses in its group. There are two common forms:

- **Item-by-item judgments**: Break the rubric into checklist items, ask the judge whether each item is satisfied (0/1), then take a weighted sum and normalize it into a score. Methods such as RaR fall into this category, as do the paper's GRPO explicit and GDPO baselines, which score individual rubric items.
- **Holistic scoring**: Ask the judge to use the rubric to assign a single score to the whole response, such as 1–10. The paper's implicit baselines fall into this category.

Both forms ultimately produce an independent score for each response and use it directly as the reward. The authors identify three problems with this approach in long-form RL:

1. **Score scales are hard to calibrate.** Holistic scoring compresses a long report into a single number. Item-by-item judgments only ask whether each requirement is met, but the judge independently decides what counts as meeting it, while the weights and aggregation method determine the final scale. A score of 0.7 need not represent the same quality across questions or training stages.
2. **Weak discrimination within a group.** The 8 responses to the same question often receive identical judgments on many checklist items: everyone satisfies the easy items and scores 1; nobody satisfies the hard items and everyone scores 0. These items provide no within-group differences. After aggregation, rewards cluster together, leaving GRPO's advantages dominated by noise. Holistic scoring has a similar problem: the 8 responses often cluster around scores of 7 or 8.
3. **Scores saturate easily.** After some training, the policy learns to satisfy the easy scoring criteria—structure, format, and coverage—which soon all score 1. Harder dimensions, such as evidence use and insight, remain poorly differentiated.

The key observation comes from GRPO itself: it only needs to know which response is better within the same question; it has no use for a globally calibrated absolute score. Given that, we can ask the judge to compare responses directly.

The comparison idea comes from TourRank, which ranks candidates through multiple rounds of within-group comparisons. The paper also includes a Pairwise baseline: compare every pair among the 8 responses (28 matches) and use the win rate as the reward, an idea drawn from ArenaRL. Tournament-GRPO instead runs an elimination tournament: form pairs, advance the winners, regroup, and continue until one candidate remains; repeat the entire tournament several times and accumulate wins.

## 3. A Plain-Language Analogy: Testing New Beans at a Pour-Over Coffee Shop

Imagine a pour-over coffee shop has received a new batch of Ethiopian beans, and 8 baristas each brew a cup. The manager has a tasting card: clear floral and fruity aromas, bright but gentle acidity, clean sweetness, and no bitter aftertaste.

With **pointwise scoring**, the manager tastes each cup and scores it independently. They can check each item on the tasting card—are the floral and fruity notes clear? Is the acidity harsh?—and calculate a weighted sum, or directly assign a score of 1–10. Either way, they consider only the cup in front of them. With a checklist, several of the 8 cups may satisfy every item and receive identical totals. With direct scoring, by the 6th cup, the manager may struggle to remember whether the floral aroma of the 1st cup deserved a 7 or an 8.

In a **tournament**, the manager tastes only two cups at a time and uses the tasting card to decide which is better. The 8 cups are drawn into 4 pairs, and the winners advance; the remaining 4 cups form 2 pairs, until 1 champion emerges. Each win earns 1 point. To reduce the luck of the draw, the whole tournament is redrawn and repeated 3 more times. The final ranking uses accumulated wins:

- The barista with the most wins receives 1, the one with the fewest receives 0, and the rest receive proportional scores in between.
- If everyone has the same number of wins, everyone receives 0.5.
- A barista who never serves a cup (fails to produce a final answer) still enters the tournament, but their final score is set to 0.
- Anyone who records the brewing process according to the shop's rules (format compliance) receives an additional 1.

The scores for these 8 cups are then compared: brewing methods that perform above average are encouraged, while those below average are discouraged. In the next round, everyone adjusts the grind size, water temperature, and pouring rhythm, then brews and compares again.

Throughout this process, the manager answers only one question: “Which of these two cups is better?” That is much easier to answer than “What score does this cup deserve?”

## 4. Key Variables and Formulas

| Symbol | Meaning | Value in this chapter |
| --- | --- | --- |
| K | Number of responses sampled per question (the GRPO group) | 8 |
| G | Number of candidates per match | 2 |
| W | Number of winners advancing from each match | 1 |
| F | Number of remaining candidates at which a tournament ends | 1 |
| M | Number of repeats of the entire tournament | 4 |
| $s_i$ | Accumulated wins for response $i$ across M tournaments | 0–12 |

With K=8, one tournament has 4 + 2 + 1 = 7 matches; repeating it 4 times gives 28 matches. Accumulated wins are normalized into tournament rewards:

```math
r_i^{\text{tour}} = \frac{s_i - \min_j s_j}{\max_j s_j - \min_j s_j}
```

When all $s_i$ are equal, $r_i^{\text{tour}} = 0.5$; responses without a final answer are set to 0 after normalization. The total reward adds a 0/1 format reward, giving a range of 0–2:

```math
r_i = r_i^{\text{tour}} + r_i^{\text{fmt}}
```

Following the authors' implementation, the advantage uses the within-group sample standard deviation (`ddof=1`):

```math
A_i = \frac{r_i - \bar{r}}{s_r + 10^{-6}}
```

Here, $\bar{r}$ is the mean reward within the group, and $s_r$ is the sample standard deviation computed with $K-1$ in the denominator.

Below is a real scored group from training, for the materials science question at the start of this chapter:

![A real Tournament-GRPO scoring group](./images/tournament-real-group.png)

A few details in the figure are worth examining:

- The final of tournament 1 was #4 against #5, and #5 won. Across the 4 tournaments, #5 won 9 matches and #4 won 7, so #5's tournament reward was 1.0 and #4's was 7/9 ≈ 0.78.
- #4 and #7 complied with the format and each received an extra 1, pushing their total rewards above #5's. In this group, the format reward affected the ranking more than the tournament reward did.
- #1 had no answer but still won 2 matches. It was selected as the winner between two empty answers. Its reward was still set to 0 after normalization, but those 2 matches consumed judge calls all the same.

## 5. How Did We Reproduce It?

### 5.1 The Training Loop

At each step, `train.py` does the following:

```text
1. 把最新权重交给 sampler。
2. 取 8 道题，每题采样 8 条多轮轨迹（rollout.py）。
3. 对每道题：格式检查 + 28 场淘汰赛，得到 8 个 reward（reward.py + judge.py）。
4. 组内标准化得到 advantage。
5. 用自定义 PPO loss 做一次 forward_backward，再做一次 AdamW。
```

Each step contains 64 trajectories and 224 judge comparisons, with only one update per batch.

### 5.2 Data

The training data comes directly from the Parquet files published in the authors' repository: 4,627 DR-Tulu examples, 2,793 RaR-Science examples, and 960 RaR-Medicine examples, totaling 8,380 training questions and 441 validation questions. Each question includes a set of rubrics, with a title, description, dimension, and importance (critical / important, etc.) for each item.

`prepare_data.py` only converts the format and filters by length: following the authors' verl configuration, it removes prompts longer than 4096 tokens. Filtering with the Qwen3.5-4B tokenizer leaves 8,378 / 440 examples. Evaluation uses the Deep Research Bench included by the authors: 100 research questions with scoring criteria and reference reports.

### 5.3 Rollouts: Two Tools and One XML Protocol

The model has two tools, with the same names as in the authors' implementation: `google_search` returns search snippets, and `browse_webpage` reads webpage content. A trajectory looks like this (an abridged excerpt from #4 in the group above):

```text
<think>I need to research the progress made in data mining applications in materials science ...</think>
<call_tool name="google_search">data mining applications materials science research progress 2023 2024</call_tool>
<tool_output>
<snippet>Title: Recent progress on machine learning with limited materials data ... </snippet>
...
Environment notice: tool budget exhausted; no tool calls remain. ...
</tool_output>
<think>Based on all the evidence collected, ...</think>
<answer>Data mining has made significant progress in materials science research ...</answer>
```

The contents of `<tool_output>` come from the external environment. They remain in the context, but their mask is 0, so they do not contribute to the loss. The model is responsible only for the think blocks, tool calls, and answers it generates itself. This follows the same approach as the Search-R1 chapter.

To reduce the reproduction cost, I changed search from Top 10 to Top 3 and reduced the interaction budget from 8 turns / 7 tool calls to 6 turns / 4 tool calls. With the shorter budget, the model often continued researching at its original pace and was cut off when requesting a 5th tool call, without writing a single word of its answer. If all 8 trajectories in a group have empty answers, both rewards and advantages are 0, and that step produces no gradient.

I therefore added an environment notice at the end of the 4th tool result: the tool budget is exhausted; write the final answer directly from the available evidence, and state the limitations if that evidence is insufficient. Once the model knows it needs to wrap up, the group can contain differences between completed and missing answers, and between better and worse reports.

This notice is environment content and also has a mask of 0. It is a change I added to the authors' setup and is disclosed in the differences table.

### 5.4 Reward: The Tournament Itself

The core tournament code is `run_tournament` in `reward.py`. With logging removed, it takes only these lines:

```python
scores = [0.0] * len(answers)
rng = random.Random(judge.seed)
for repeat in range(TOURNAMENT_REPEATS):           # M = 4
    candidates = list(range(len(answers)))
    while len(candidates) > 1:                       # F = 1
        rng.shuffle(candidates)
        num_groups = len(candidates) // PLAYERS_PER_MATCH   # G = 2
        groups = [candidates[i::num_groups] for i in range(num_groups)]
        winners = []
        for group in groups:
            local_index, _ = judge.judge(query, rubrics, [answers[i] for i in group])
            winner = group[local_index]              # W = 1
            scores[winner] += 1.0
            winners.append(winner)
        candidates = winners
```

The judge sees only the question, the complete rubric, and the final `<answer>` from each response. It does not see the reasoning or search process. The authors' judge prompt asks it to “select exactly one winner, preferring responses that are accurate, complete, well-supported, and directly answer the question.”

The format reward is 0 or 1 and checks the entire trajectory: it must begin with a non-empty think block, include at least one search, follow every tool call with `tool_output` and think, and end with exactly one non-empty answer. Stray text outside the tags and malformed XML are not allowed. These rules are copied from the authors' `tournament_grpo_reward.py`.

When a judge request fails, the authors' implementation randomly selects a winner and continues training. I removed this fallback: training stops after three failures. This decision came from an actual incident. At step 15 of a 20-step run with the old configuration, repeated network failures at the judge endpoint caused 150 of the 224 matches to fall back to random winners, making that step's rewards largely random. Updating weights with random rewards is worse than stopping, fixing the endpoint, and resuming training.

### 5.5 Advantage and PPO Loss

The advantage uses the group standardization from Section 4. The loss implements the authors' vanilla PPO as a PyTRIO custom loss (`ppo_objective` in `train.py`):

```python
log_ratio = (current - old).clamp(-20.0, 20.0)
ratio = log_ratio.exp()
losses1 = -advantages * ratio
losses2 = -advantages * ratio.clamp(0.8, 1.2)          # clip 0.2
clipped = torch.maximum(losses1, losses2)
dual_bound = -advantages * 3.0                         # dual clip 3，只作用于负 advantage
losses = torch.where(advantages < 0, torch.minimum(dual_bound, clipped), clipped)
loss = losses.mean()                                   # 整个 batch 的 assistant token 取均值
```

There is no KL term or entropy term. Each batch is updated only once, so the old logprob is taken directly from `current.detach()` in the training-side forward pass. The ratio is therefore always 1 and clipping never activates, but the gradient remains $-A \cdot \nabla \log \pi$, equivalent to the authors' setup. If the same batch is updated multiple times in the future, old must be fixed before the first update.

### 5.6 Differences from the Authors' Implementation

| Item | Authors | This chapter | Impact |
| --- | --- | --- | --- |
| Policy | Qwen2.5-7B-Instruct, full-parameter training, lr 1e-6 | Qwen3.5-4B, LoRA rank 32, lr 4e-5 | Absolute scores cannot be compared with the paper |
| Judge | Self-hosted Qwen2.5-72B-Instruct, chat prompt | Jev `jev-1.14.0`, prompt adapted to state + a single-choice question | Different judge preferences; see Section 8 |
| Search | Search service with an unspecified provider, Top 10 | DeepSeek Search API, Top 3; queries truncated beyond 128 characters | Less evidence coverage; results change over time |
| Browsing | Jina Reader | Jina Reader; the server uses the official mainland China domain `r.jinaai.cn` | Different endpoint, same protocol |
| Interaction budget | 8 turns / 7 tool calls, stops immediately when exhausted | 6 turns / 4 tool calls, with a wrap-up notice appended to the last tool result | Shallower research; added environment notice |
| Judge failure | Select a random winner | Stop training and resume from a checkpoint | No updates using random rewards |
| Training scale | Batch 16, multiple epochs, approximately 1,100–1,600 steps | Batch of 8 questions × 8 trajectories, 20 steps | Only early behavior changes can be observed |
| Experiment tracking | WandB | SwanLab | None |

The tournament structure (G=2, W=1, F=1, M=4), normalization, setting missing answers to 0, format rules, 18000-token response budget, 3000-character tool truncation, and the authors' prompt are all unchanged.

## 6. Experimental Results

### 6.1 20-Step Training

Configuration: Qwen3.5-4B, LoRA rank 32, lr 4e-5, 8 questions × 8 trajectories per step, 20 steps in total. Full records are available on [SwanLab](https://swanlab.cn/@MRLE/tournament-grpo/v1/x1tb4s/runs/s474uc49/chart).

![Reward curves over 20 training steps](./images/swanlab_reward.png)

| Metric | Step 1 | Step 10 | Step 20 |
| --- | ---: | ---: | ---: |
| Answer completion rate | 64.1% | 93.8% | 96.9% |
| Format compliance rate | 20.3% | 65.6% | 45.3% |
| Total reward | 0.489 | 1.022 | 0.794 |
| Tournament reward | 0.286 | 0.365 | 0.340 |
| Search calls per trajectory | 2.58 | 3.17 | 2.47 |
| Browsing calls per trajectory | 0.73 | 0.70 | 0.47 |
| Response tokens per trajectory (including tool results) | 2,778 | 3,556 | 2,725 |

Answer completion improved most consistently: its mean over the first 3 steps was 61.5%, versus 95.3% over the last 3. Format compliance improved more slowly and fluctuated more, with the corresponding 3-step means rising from 19.3% to 52.6%; it reached 65.6% at step 10, then fell back to 45.3% at step 20. Tournament reward rose only from 0.29 to 0.32. `reward/degenerate_group_rate` stayed at 0 throughout, so every group received nonzero advantages.

![Rollout curves over 20 training steps](./images/swanlab_rollout.png)

`rollout/finish_no_tool_call` rose from 0.78 to almost 1. In the first few steps, approximately 22% of trajectories were cut off while requesting a 5th tool call. This had largely disappeared after step 8: the model learned to wrap up after receiving the budget notice. Search calls, browsing calls, and response length fluctuated over the 20 steps without a clear trend.

### 6.2 Eval

Evaluation uses all 100 questions in the authors' bundled Deep Research Bench. Base and Step 20 use the same tool configuration, greedily generating 1 trajectory per question (seed 42, Top 3 search, at most 4 tool calls).

Scoring uses RACE: for each question's criteria, the judge compares the evaluated report with the authors' reference report item by item, assigns each a score of 0–10, and applies the weights. The final score is the evaluated report's share of the sum of the two scores; 50 means a tie with the reference report. The judge is `deepseek-flash` (thinking disabled), whereas the paper uses Qwen2.5-72B-Instruct.

| RACE (0–100) | Base | Step 20 | Difference |
| --- | ---: | ---: | ---: |
| Comprehensiveness | 19.21 | 32.33 | +13.12 |
| Insight | 17.11 | 29.67 | +12.56 |
| Instruction Following | 24.95 | 40.77 | +15.82 |
| Readability | 23.91 | 40.01 | +16.10 |
| **Overall** | **20.65** | **34.70** | **+14.05** |

| Generation metric | Base | Step 20 |
| --- | ---: | ---: |
| Answer completed | 60/100 | 98/100 |
| Format compliant | 42/100 | 72/100 |
| Tool calls per question | 3.98 | 3.94 |
| Model-generated tokens per question | 826 | 1,641 |

Most of the gain comes from fewer blank reports. Base exhausted its tool budget without writing a report on 40 questions, which scored 0; Step 20 did so on only 2. On the 59 questions where both models produced reports, Overall rose only from 34.49 to 35.25, indicating similar quality for individual reports. The doubling of generated tokens has the same explanation: Base's blank reports averaged only 220 tokens.

SwanLab: [Base generation](https://swanlab.cn/@MRLE/tournament-grpo/runs/5cqbugvp), [Step 20 generation](https://swanlab.cn/@MRLE/tournament-grpo/runs/0fw66y62), [Base RACE](https://swanlab.cn/@MRLE/tournament-grpo/runs/5lz7d3t0), [Step 20 RACE](https://swanlab.cn/@MRLE/tournament-grpo/runs/7gyzdjge).

### 6.3 Cost

| Stage | Item | Cost (yuan) |
| --- | --- | ---: |
| Training | PyTRIO (training and sampling) | 60.96 |
| Training | DeepSeek Search | 16.96 |
| Training | Jev judge | 1.00 |
| Evaluation | PyTRIO (sampling) | 3.20 |
| Evaluation | DeepSeek | 9.92 |
| | **Total** | **92.04** |

PyTRIO bill:

![PyTRIO bill](./images/pytrioBill.png)

Training took 4 hours 3 minutes in total, averaging 12.2 minutes and approximately 3.95 yuan per step. Judging accounted for most of the time: 4,480 comparisons took approximately 2 hours 22 minutes, or 58% of the total, yet cost only 1 yuan. Across 20 steps, the judge read approximately 7.58 million input tokens and produced approximately 139,000 output tokens; the model made 3,218 searches and browsed 722 webpages. During evaluation, generating 100 reports took Base 4.5 minutes and Step 20 13.2 minutes.

## 7. Which Metrics Matter Most During Training?

Tournament rewards measure relative position within a group, not absolute quality, so the reward curve cannot serve as a convergence indicator. Model quality requires independent evaluation. During training, focus on these groups of metrics:

| Metric | What to look for |
| --- | --- |
| `reward/has_answer`, `reward/format` | Answer completion and format compliance rates |
| `reward/tournament` | Mean tournament reward; missing answers score 0, so it rises with the answer completion rate |
| `reward/degenerate_group_rate` | Fraction of questions whose 8 rewards are identical, producing no gradient |
| `rollout/finish_*` | Whether trajectories end normally or are cut off by turn, tool-call, or length limits |
| `rollout/search_calls`, `rollout/browse_calls`, `rollout/response_tokens` | Tool usage and trajectory length |
| `tools/error_rate` | Whether search and browsing are stable; environment errors directly contaminate rewards |
| `judge/input_tokens` | Judge token usage, which directly determines cost |

To inspect the judge, sample records from `judge.jsonl`. It contains the candidates, winner, and raw judge output for every match.

## 8. What Are the Limits of This Reproduction?

- The policy, judge, search backend, and training scale all differ from the paper, so the scores cannot be directly compared. This chapter checks whether the training pipeline works after connecting tournament rewards to PyTRIO, and how early behavior changes.

## 9. Insight and Action

The following observations emerged during reproduction and have not yet been tested.

- **Repeating a binary tournament 4 times does not save judge calls.** With K=8, 4 × 7 = 28 matches, the same as the C(8,2) exhaustive pairwise comparisons; the difference is how rewards are assigned. The paper's configurations that save calls use 2–3 repeats. Action: run M=2 and M=4 offline on the same batch of trajectories and compare the advantage rankings.

- **Early reward gains come mainly from the format reward.** From the first 3 steps to the last 3, total reward rose from 0.485 → 0.849, with approximately 92% of the increase coming from format reward (0.193 → 0.526); tournament reward rose only from 0.293 to 0.323. Action: track has_answer, format, and tournament separately, then examine within-group tournament reward variance once answer completion stabilizes.

## 10. Summary

Tournament-GRPO makes a small change: it replaces the source of the reward.

```text
1. 每道题采样 8 份带工具调用的研究报告。
2. 两两比赛、胜者晋级，整场重复 4 次，累计胜场。
3. 组内 min-max 归一化，无答案记 0，再加 0/1 格式奖励。
4. 组内标准化得到 advantage，用 vanilla PPO 更新。
```

The reasoning behind it is that GRPO only needs to know which response is better within the group, so we can ask the judge directly. This is a natural approach for long-form tasks without standard answers. The cost is that judge calls grow linearly with M, and reward quality depends entirely on the judge.

The environment took the most time during reproduction: cutting the tool budget left the model unable to wrap up, judge network failures produced random rewards, and the search endpoint imposed a query-length limit. These issues are separate from the algorithm itself, yet determine whether a training step produces useful gradients. In Agentic RL, environment quality is itself an experimental variable.

This 20-step run cost 78.92 yuan and took 4 hours 3 minutes; evaluation added 13.12 yuan, for a total of 92.04 yuan. Across all 100 Deep Research Bench questions, answer completion rose from 60% → 98%, format compliance from 42% → 72%, and RACE Overall from 20.65 → 34.70.

## References

1. Zixuan Yang et al. [Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation](https://arxiv.org/abs/2605.26958v2), EMNLP 2026.
2. [PuffYang/Tournament_GRPO@eea826cf](https://github.com/PuffYang/Tournament_GRPO/tree/eea826cf66705cac9ec81823f746d0ccd6074228): authors' code, training data, and Deep Research Bench
3. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
4. [DeepSeek Search API](https://api-docs.deepseek.com/zh-cn/guides/search/)
5. [Jina Reader](https://jina.ai/reader/)
6. [PyTrio-Jev chapter](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/pytrio-jev/readme.md)
7. [Complete Tournament-GRPO PyTRIO code](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/tournament-grpo)
8. [PyTRIO documentation](https://docs.pytrio.com/docs) / [Custom loss](https://docs.pytrio.com/docs/guide/custom_loss)
9. [SwanLab documentation](https://docs.swanlab.cn/)
