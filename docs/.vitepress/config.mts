import { defineConfig } from 'vitepress'
import { slug } from 'github-slugger'
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { localizedChapters, repoUrl, chapterLink } from '../chapters.mjs'
import { siteUrl, siteBase } from '../site.mjs'
import { addSeo, pageDates } from './seo.mjs'
import { imageSize, imageHints } from './images.mjs'

// Serve every page's raw markdown next to its HTML so readers can copy it
// or hand the URL to an LLM (mirrors the page path, e.g. /experiments/grpo/index.md).
// A UTF-8 BOM keeps browsers from misdetecting the encoding when the
// static host serves .md without a charset header.
function copyRawMarkdown(srcDir: string, outDir: string) {
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) { if (entry !== 'public') walk(full); continue }
      if (entry.endsWith('.md')) {
        const dest = join(outDir, relative(srcDir, full))
        mkdirSync(dirname(dest), { recursive: true })
        writeFileSync(dest, '﻿' + readFileSync(full, 'utf8'))
      }
    }
  }
  walk(srcDir)
}

function navigation(locale: 'zh' | 'en') {  const en = locale === 'en'
  const prefix = en ? '/en' : ''
  const chapters = localizedChapters(locale)
  return {
    nav: [
      { text: en ? 'Home' : '首页', link: `${prefix}/` },
      { text: en ? 'Experiments' : '实验手册', link: `${prefix}/experiments/`, activeMatch: `${prefix}/experiments/` },
      { text: en ? 'Learning path' : '学习路线', link: `${prefix}/guide/learning-path.html` },
      { text: en ? 'About' : '关于这个 Lab', link: `${prefix}/guide/introduction.html` },
    ],
    sidebar: [
      { text: en ? 'Start here' : '从这里开始', items: [
        { text: en ? 'About this lab' : '关于这个 Lab', link: `${prefix}/guide/introduction.html` },
        { text: en ? 'Learning path' : '学习路线', link: `${prefix}/guide/learning-path.html` },
        { text: en ? 'Environment setup' : '环境准备', link: `${prefix}/guide/quickstart.html` },
        { text: en ? 'GRPO, DAPO & GSPO' : 'GRPO、DAPO 与 GSPO', link: `${prefix}/guide/grpo-dapo-gspo.html` },
        { text: en ? 'Contribute' : '参与贡献', link: `${prefix}/guide/contributing.html` },
        { text: en ? 'All experiments' : '全部实验', link: `${prefix}/experiments/` },
      ] },
      { text: en ? 'Training platforms' : '训练平台', items: [
        { text: en ? 'PyTRIO tutorials' : 'PyTRIO 实战教程', link: `${prefix}/guide/pytrio.html` },
        { text: en ? 'Tinker & PyTRIO workflows' : 'Tinker 与 PyTRIO', link: `${prefix}/guide/tinker-pytrio.html` },
      ] },
      {
        text: en ? 'Experiment reading order' : '实验阅读顺序', collapsed: false,
        items: chapters.map(c => ({
          text: `${c.number}  ${c.title}`, link: chapterLink(c, locale),
          ...(c.start ? { collapsed: true, items: [{ text: en ? 'Quick start' : '快速启动', link: `${chapterLink(c, locale)}run.html` }] } : {}),
        })),
      },
    ],
    outline: { label: en ? 'On this page' : '本页目录', level: [2, 3] as [number, number] },
    docFooter: { prev: en ? 'Previous page' : '上一篇', next: en ? 'Next page' : '下一篇' },
    darkModeSwitchLabel: en ? 'Appearance' : '外观',
    lightModeSwitchTitle: en ? 'Switch to light theme' : '切换至浅色',
    darkModeSwitchTitle: en ? 'Switch to dark theme' : '切换至深色',
    sidebarMenuLabel: en ? 'Menu' : '章节目录', returnToTopLabel: en ? 'Back to top' : '回到顶部',
    langMenuLabel: en ? 'Change language' : '切换语言', skipToContentLabel: en ? 'Skip to content' : '跳到正文',
    editLink: {
      pattern: ({ frontmatter, relativePath }) => {
        // VitePress serializes this function for the browser; keep it self-contained.
        const repository = 'https://github.com/datawhalechina/agentic-rl-lab'
        return frontmatter.translationSource
          ? `${repository}/edit/doc/${frontmatter.translationSource}`
          : frontmatter.source ? `${repository}/edit/main/${frontmatter.source}` : `${repository}/edit/doc/docs/content/${relativePath}`
      },
      text: en ? 'Edit this page on GitHub' : '在 GitHub 上编辑此页',
    },
    notFound: { title: en ? 'This note is missing' : '这页笔记还没找到', quote: en ? 'Try another path into reinforcement learning.' : '换一个入口，继续探索强化学习。', linkLabel: en ? 'Go home' : '返回首页', linkText: en ? 'Go home' : '返回首页' },
    footer: { message: en ? 'Open code. Real experiments. Honest limitations.' : '开放代码，记录实验，也记录边界。', copyright: 'Agentic RL Lab · Built with curiosity.' },
  }
}

export default defineConfig({
  title: 'Agentic RL Lab',
  description: '基于 PyTRIO 的 LLM 与 Agent 强化学习实战笔记，涵盖 GRPO、DAPO、GSPO、工具调用、多模态训练与 Tinker 工作流对照。',
  lang: 'zh-CN', base: siteBase, srcDir: './content', cleanUrls: false, lastUpdated: false,
  locales: {
    root: { label: '简体中文', lang: 'zh-CN', themeConfig: navigation('zh') },
    en: { label: 'English', lang: 'en', description: 'Hands-on LLM and agent reinforcement learning with PyTRIO, including GRPO, DAPO, GSPO, tool use, multimodal training and Tinker workflow notes.', themeConfig: navigation('en') },
  },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${siteBase}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#fdfbf9' }],
    ['meta', { name: 'msvalidate.01', content: '14E8835C29152AB0D1C6C338FE11ACEB' }],
    ['meta', { name: 'google-site-verification', content: 'plkZ1LHW27AE_IOw7VfWvPkWazMXaKAhksRUJMs92DE' }],
  ],
  transformPageData: addSeo,
  buildEnd: (siteConfig) => copyRawMarkdown(siteConfig.srcDir, siteConfig.outDir),
  vite: {
    plugins: [{
      name: 'raw-markdown-charset',
      // The dev server bypasses buildEnd; give source .md responses an
      // explicit charset so Chinese text does not render as mojibake.
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.includes('.md')) res.setHeader('Content-Type', 'text/markdown; charset=utf-8')
          next()
        })
      },
    }],
  },
  // VitePress's generated 404 bypasses transformPageData.
  transformHead: ({ pageData }) => pageData.relativePath === '404.md'
    ? [['meta', { name: 'robots', content: 'noindex' }]] : [],
  sitemap: {
    hostname: siteUrl,
    transformItems: items => items.map(item => ({ ...item, lastmod: pageDates.get(item.url) })),
  },
  markdown: {
    math: true, image: { lazyLoading: true }, anchor: { slugify: slug },
    theme: { light: 'github-light-high-contrast', dark: 'github-dark' },
    config(md) {
      const renderImage = md.renderer.rules.image!
      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const size = imageSize(tokens[idx].attrGet('src'))
        if (size) {
          tokens[idx].attrSet('width', String(size.width))
          tokens[idx].attrSet('height', String(size.height))
        }
        tokens[idx].attrSet('decoding', 'async')
        return renderImage(tokens, idx, options, env, self)
      }
      const block = md.renderer.rules.html_block!
      const inline = md.renderer.rules.html_inline!
      for (const [type, renderer] of [['html_block', block], ['html_inline', inline]] as const) {
        md.renderer.rules[type] = (tokens, idx, options, env, self) => imageHints(renderer(tokens, idx, options, env, self))
          .replace(/\bsrc="(\/media\/[^\"]+)"/g, (_, url) => `:src="'${siteBase.slice(0, -1) + url}'"`)
      }
    },
  },
  themeConfig: {
    logo: { src: '/favicon.svg', width: 30, height: 30 }, siteTitle: 'Agentic RL Lab',
    socialLinks: [{ icon: 'github', link: repoUrl, ariaLabel: 'GitHub' }],
    search: {
      provider: 'local',
      options: {
        _render(src, env, md) {
          const html = md.render(src, env)
          if (env.frontmatter?.search === false) return ''
          return env.frontmatter?.chapter
            ? `<h1 id="article-title">${md.utils.escapeHtml(env.frontmatter.title)}<a href="#article-title" class="header-anchor">#</a></h1><p>${md.utils.escapeHtml(env.frontmatter.description)}</p>${html}`
            : html
        },
        miniSearch: { options: {
          tokenize: text => Array.from(new Intl.Segmenter('zh', { granularity: 'word' }).segment(text), part => part.segment).filter(word => /[\p{L}\p{N}]/u.test(word)),
        } },
        locales: { root: { translations: {
          button: { buttonText: '搜索实验与笔记', buttonAriaLabel: '搜索实验与笔记' },
          modal: {
            displayDetails: '显示详情', resetButtonTitle: '清空搜索', backButtonTitle: '关闭搜索', noResultsText: '没有找到相关内容',
            footer: { selectText: '选择', selectKeyAriaLabel: '回车', navigateText: '切换', navigateUpKeyAriaLabel: '向上', navigateDownKeyAriaLabel: '向下', closeText: '关闭', closeKeyAriaLabel: 'Esc' },
          },
        } } },
      },
    },
  },
})
