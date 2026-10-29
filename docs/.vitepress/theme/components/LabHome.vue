<script setup>
import { withBase } from 'vitepress'
import { repoUrl, chapterLink } from '../../../chapters.mjs'
import { computed, ref } from 'vue'
import { useLocale } from '../composables.js'
import LabIcon from './LabIcon.vue'
import TrainingLoop from './TrainingLoop.vue'
const { t, locale, chapters, groups, path } = useLocale()
const activeGroup = ref('all')
const visibleChapters = computed(() => activeGroup.value === 'all' ? chapters.value : chapters.value.filter(c => c.group === activeGroup.value))
const groupLabel = (id) => groups.value.find(g => g.id === id)?.label
const groupCount = (id) => chapters.value.filter(c => c.group === id).length
const tilts = [-1.1, 0.7, -0.6, 1.0, -0.9, 0.5, -1.2, 0.8]
const tilt = (i) => `${tilts[i % tilts.length]}deg`
</script>

<template>
  <main class="lab-home">
    <section class="lab-hero lab-width">
      <div class="hero-copy">
        <p class="eyebrow"><span class="status-dot" /> AN OPEN RESEARCH NOTEBOOK</p>
        <h1>{{ t('从一篇论文，', 'From a paper, ') }}<br>{{ t('到一次', 'to a real ') }}<span class="hero-emphasis">{{ t('真实的训练', 'training run') }}<span class="hero-period">{{ t('。', '.') }}</span></span></h1>
        <p class="hero-description">{{ t('用 PyTRIO 实践 LLM 与 Agent 强化学习。', 'Hands-on LLM and agent reinforcement learning with PyTRIO. ') }}<br>{{ t('15 篇可运行的实验笔记：从 GRPO / DAPO / GSPO，到 Search-R1、ReTool 与多模态 Agent。', '15 runnable experiment notes: from GRPO, DAPO and GSPO to Search-R1, ReTool and multimodal agents.') }}</p>
        <div class="hero-actions">
          <a class="lab-button primary" :href="withBase(path('/guide/quickstart.html'))">{{ t('开始第一个实验', 'Run your first experiment') }} <LabIcon name="arrow" :size="17" /></a>
          <a class="lab-button secondary" :href="withBase(path('/experiments/'))">{{ t('浏览实验手册', 'Browse experiments') }} <LabIcon name="book" :size="17" /></a>
        </div>
        <div class="hero-quickstart"><code>git clone datawhalechina/agentic-rl-lab && uv sync</code><a :href="withBase(path('/guide/quickstart.html'))">{{ t('安装与入门', 'Quickstart') }} <LabIcon name="arrow" :size="13" /></a></div>
      </div>
      <TrainingLoop />
    </section>

    <div class="lab-proof lab-width">
      <div class="proof-intro"><span class="small-label">START HERE</span><p>{{ t('新读者从', 'New here? Start with') }} <a :href="withBase(path('/experiments/loss-functions/'))">Loss Functions</a> {{ t('和', 'and') }} <a :href="withBase(path('/experiments/grpo/'))">GRPO</a> {{ t('读起。', '.') }}</p></div>
      <div class="proof-number"><strong>{{ chapters.length.toString().padStart(2, '0') }}</strong><span>{{ t('篇实验笔记', 'experiment notes') }}</span></div>
      <div class="proof-number"><strong>04</strong><span>{{ t('个探索方向', 'research tracks') }}</span></div>
      <div class="proof-tools"><span class="small-label">BUILT WITH</span><div><a href="https://pytrio.com" target="_blank" rel="noreferrer">PyTRIO <span>↗</span></a><i /> <a href="https://swanlab.cn" target="_blank" rel="noreferrer">SwanLab <span>↗</span></a></div></div>
    </div>

    <section class="home-section board-section lab-width">
      <div class="section-top"><div><p class="eyebrow">THE BOARD</p><h2>{{ t('所有实验都在这里～', 'Every experiment, right here.') }}</h2></div><a class="text-link" :href="withBase(path('/guide/learning-path.html'))">{{ t('查看学习路线', 'Find your learning path') }} <LabIcon name="arrow" :size="17" /></a></div>
      <div class="board-tabs">
        <button type="button" :class="{ active: activeGroup === 'all' }" @click="activeGroup = 'all'">{{ t('全部', 'All') }} {{ chapters.length }}</button>
        <button v-for="group in groups" :key="group.id" type="button" :class="{ active: activeGroup === group.id }" @click="activeGroup = group.id"><span class="tab-number">{{ group.number }}</span>{{ group.label }} {{ groupCount(group.id) }}</button>
      </div>
      <div class="note-wall">
        <a v-for="(chapter, i) in visibleChapters" :key="chapter.id" :href="withBase(chapterLink(chapter, locale))" class="sticky-note" :style="{ '--tilt': tilt(i) }">
          <span class="note-tape" aria-hidden="true" />
          <div class="note-head"><span>LAB {{ chapter.number }}</span><span>{{ groupLabel(chapter.group) }}</span></div>
          <h3>{{ chapter.title }}</h3>
          <p class="note-subtitle">{{ chapter.subtitle }}</p>
          <p class="note-desc">{{ chapter.description }}</p>
          <div class="note-foot"><span v-for="tag in chapter.tags" :key="tag">{{ tag }}</span><span class="read-label">{{ t('阅读实验', 'Read') }} <LabIcon name="arrow" :size="13" /></span></div>
        </a>
      </div>
    </section>

    <section class="home-section lab-width">
      <div class="section-top"><div><p class="eyebrow">TRAINING PLATFORMS</p><h2>{{ t('从训练工具，走进实验。', 'From training tools to experiments.') }}</h2></div></div>
      <div class="platform-guides">
        <a class="platform-guide sticky-note wide-note" :href="withBase(path('/guide/pytrio.html'))"><span class="note-tape" aria-hidden="true" /><h3>{{ t('PyTRIO 实战教程', 'PyTRIO tutorials') }} <LabIcon name="arrow" :size="18" /></h3><p>{{ t('从安装与 GRPO 开始，找到工具调用、蒸馏和多模态训练的代码入口。', 'Start with setup and GRPO, then explore code for tool use, distillation and multimodal training.') }}</p></a>
        <a class="platform-guide sticky-note wide-note" :href="withBase(path('/guide/tinker-pytrio.html'))"><span class="note-tape" aria-hidden="true" /><h3>{{ t('Tinker 与 PyTRIO', 'Tinker and PyTRIO') }} <LabIcon name="arrow" :size="18" /></h3><p>{{ t('对照采样、奖励与策略更新，理解两个训练工作流以及迁移时要检查的细节。', 'Compare sampling, rewards and policy updates, with practical checks for adapting an experiment.') }}</p></a>
      </div>
    </section>

    <section class="lab-invitation lab-width"><div class="invitation-mark"><LabIcon name="code" :size="29" /></div><div><p class="eyebrow">LEARN IN THE OPEN</p><h2>{{ t('实验会有边界，好奇心可以更远。', 'Document the limits. Keep asking questions.') }}</h2><p>{{ t('这里记录实现、结果和没跑通的地方。欢迎带着问题、复现记录和新想法一起加入。', 'Implementations, results and things that did not work. Bring your questions, reproduction notes and ideas.') }}</p></div><a class="lab-button secondary" :href="repoUrl" target="_blank" rel="noreferrer">{{ t('在 GitHub 交流', 'Discuss on GitHub') }} <LabIcon name="diagonal" :size="16" /></a></section>
  </main>
</template>
