import { execFileSync } from 'node:child_process'
import { statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { author, pageUrl, siteUrl, routePath } from '../site.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
const dates = new Map()
export const pageDates = new Map()
export function sourceDates(file) {
  if (dates.has(file)) return dates.get(file)
  const history = execFileSync('git', ['log', '--follow', '--format=%aI', '--', file], { cwd: root, encoding: 'utf8' }).trim().split('\n').filter(Boolean)
  const fallback = statSync(root + file).mtime.toISOString()
  const result = { published: history.at(-1) || fallback, modified: history[0] || fallback }
  dates.set(file, result)
  return result
}

export function addSeo(page) {
  const path = page.relativePath
  if (path === '404.md') {
    page.frontmatter.head = [...(page.frontmatter.head || []), ['meta', { name: 'robots', content: 'noindex' }]]
    return
  }
  const en = path.startsWith('en/')
  const original = path.replace(/^en\//, '')
  const url = pageUrl(path)
  const zhUrl = pageUrl(original)
  const enUrl = pageUrl('en/' + original)
  const title = page.title + ' | Agentic RL Lab'
  const description = page.description
  const image = siteUrl + 'social-card.png'
  const { published, modified } = page.frontmatter.sourceDates || sourceDates('docs/content/' + path)
  pageDates.set(routePath(path), modified)
  const isArticle = page.frontmatter.layout !== 'page'
  const pageAuthor = page.frontmatter.author || author
  const schema = isArticle ? {
    '@context': 'https://schema.org', '@type': 'BlogPosting', '@id': url + '#article',
    headline: page.title, description, inLanguage: en ? 'en' : 'zh-CN',
    url, mainEntityOfPage: url, image: [image],
    author: { '@type': 'Person', ...pageAuthor }, datePublished: published, dateModified: modified,
    isPartOf: { '@type': 'WebSite', name: 'Agentic RL Lab', url: en ? siteUrl + 'en/' : siteUrl },
  } : { '@context': 'https://schema.org', '@type': 'WebSite', name: 'Agentic RL Lab', url: en ? siteUrl + 'en/' : siteUrl, inLanguage: en ? 'en' : 'zh-CN', description }
  const breadcrumbs = {
    '@context': 'https://schema.org', '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: en ? 'Home' : '首页', item: en ? siteUrl + 'en/' : siteUrl },
      ...(page.frontmatter.chapter ? [{ '@type': 'ListItem', position: 2, name: en ? 'Experiments' : '实验手册', item: siteUrl + (en ? 'en/' : '') + 'experiments/' }] : []),
      { '@type': 'ListItem', position: page.frontmatter.chapter ? 3 : 2, name: page.title, item: url },
    ],
  }
  const json = (data) => JSON.stringify(data).replace(/</g, '\\u003c')
  page.frontmatter.author = pageAuthor.name
  page.frontmatter.authorUrl = pageAuthor.url
  page.frontmatter.modified = modified.slice(0, 10)
  page.frontmatter.head = [
    ...(page.frontmatter.head || []),
    ['link', { rel: 'canonical', href: url }],
    ...[['zh-CN', zhUrl], ['en', enUrl], ['x-default', zhUrl]].map(([hreflang, href]) => ['link', { rel: 'alternate', hreflang, href }]),
    ...Object.entries({ 'og:type': isArticle ? 'article' : 'website', 'og:site_name': 'Agentic RL Lab', 'og:title': title, 'og:description': description, 'og:url': url, 'og:locale': en ? 'en_US' : 'zh_CN', 'og:locale:alternate': en ? 'zh_CN' : 'en_US', 'og:image': image, 'og:image:width': '1200', 'og:image:height': '630', 'og:image:alt': 'Agentic RL Lab — papers, code and experiments' }).map(([property, content]) => ['meta', { property, content }]),
    ...Object.entries({ 'twitter:card': 'summary_large_image', 'twitter:title': title, 'twitter:description': description, 'twitter:image': image }).map(([name, content]) => ['meta', { name, content }]),
    ['script', { type: 'application/ld+json' }, json(schema)],
    ...(isArticle ? [['script', { type: 'application/ld+json' }, json(breadcrumbs)]] : []),
  ]
}
