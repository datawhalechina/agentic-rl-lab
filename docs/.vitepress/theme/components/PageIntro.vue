<script setup>
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { chapterLink, repoUrl } from '../../../chapters.mjs'
import { useLocale } from '../composables.js'
import LabIcon from './LabIcon.vue'
import MarkdownActions from './MarkdownActions.vue'
const { frontmatter } = useData()
const { t, locale, chapters } = useLocale()
const chapter = computed(() => chapters.value.find(c => c.id === frontmatter.value.chapter))
</script>

<template>
  <header v-if="chapter" class="article-intro">
    <MarkdownActions />
    <p class="eyebrow">LAB {{ chapter.number }} <span>/</span> {{ frontmatter.group }}</p>
    <h1 id="article-title">{{ chapter.title }}<span>{{ frontmatter.isRun ? t('快速启动', 'Quick start') : chapter.subtitle }}</span></h1>
    <p class="article-description">{{ frontmatter.description }}</p>
    <div class="article-meta"><span><LabIcon name="book" :size="15" /> {{ t('约', 'About') }} {{ frontmatter.readingTime }} {{ t('分钟', 'min') }}</span><a :href="frontmatter.authorUrl" rel="author">{{ frontmatter.author }}</a><span>{{ t('更新于', 'Updated') }} <time :datetime="frontmatter.modified">{{ frontmatter.modified }}</time></span><span v-for="tag in chapter.tags" :key="tag" class="article-tag">{{ tag }}</span></div>
    <div class="article-actions">
      <a v-if="chapter.start && !frontmatter.isRun" :href="withBase(`${chapterLink(chapter, locale)}run.html`)"><LabIcon name="flask" :size="16" /> {{ t('快速启动', 'Quick start') }} <span>→</span></a>
      <a v-if="frontmatter.isRun" :href="withBase(chapterLink(chapter, locale))"><LabIcon name="book" :size="16" /> {{ t('阅读算法拆解', 'Read the method') }} <span>→</span></a>
      <a :href="`${repoUrl}/tree/main/${chapter.source.slice(0, chapter.source.lastIndexOf('/'))}`" target="_blank" rel="noreferrer"><LabIcon name="code" :size="16" /> {{ t('查看代码', 'View code') }} <span>↗</span></a>
    </div>
  </header>
  <div v-else-if="frontmatter.modified" class="guide-intro"><MarkdownActions /><p class="guide-meta"><a :href="frontmatter.authorUrl" rel="author">{{ frontmatter.author }}</a> · {{ t('更新于', 'Updated') }} <time :datetime="frontmatter.modified">{{ frontmatter.modified }}</time></p></div>
</template>
