import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { load } from 'cheerio'
import { unrenderedMarkdownImages } from './rendered-images.mjs'
import { collectMarkdownUrls } from '../../docs/.vitepress/raw-markdown.mjs'
import { chapters } from '../../docs/chapters.mjs'
import { siteUrl, siteBase } from '../../docs/site.mjs'

const dist = fileURLToPath(new URL('../../docs/.vitepress/dist/', import.meta.url))
const base = siteBase
const errors = new Set()
const documents = new Map()
async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name)
    if (entry.isDirectory()) await walk(full)
    else if (entry.name.endsWith('.html')) documents.set(full, load(await readFile(full, 'utf8')))
  }
}
await walk(dist)
const chunks = path.join(dist, 'assets/chunks')
let searchSections = 0
for (const locale of ['root', 'en']) {
  const prefix = locale === 'en' ? 'en/' : ''
  const searchFile = (await readdir(chunks)).find(name => name.startsWith(`@localSearchIndex${locale}.`) && name.endsWith('.js'))
  if (!searchFile) throw new Error(`Missing ${locale} local search index`)
  const searchIndex = JSON.parse((await import(pathToFileURL(path.join(chunks, searchFile)).href)).default)
  searchSections += searchIndex.documentCount
  for (const [key, destination] of Object.entries(searchIndex.documentIds)) {
    const url = new URL(destination, 'https://docs.local')
    const file = path.join(dist, decodeURIComponent(url.pathname.slice(base.length)), url.pathname.endsWith('/') ? 'index.html' : '')
    const target = documents.get(file)
    if (!target) errors.add(`Search links to missing page: ${destination}`)
    else if (url.hash && !target('[id]').toArray().some(el => target(el).attr('id') === decodeURIComponent(url.hash.slice(1)))) errors.add(`Search links to missing anchor: ${destination}`)
    if (!searchIndex.storedFields[key]?.title) errors.add(`Search result lacks title: ${destination}`)
    if (url.pathname.startsWith(base + 'en/') !== (locale === 'en')) errors.add(`Search index mixes languages: ${destination}`)
  }
  for (const chapter of chapters) {
    const file = path.join(dist, prefix, 'experiments', chapter.id, 'index.html')
    if (!documents.has(file)) errors.add(`Missing ${locale} chapter: ${chapter.id}`)
    if (chapter.start && !documents.has(path.join(dist, prefix, 'experiments', chapter.id, 'run.html'))) errors.add(`Missing ${locale} quickstart: ${chapter.id}`)
    if (!Object.values(searchIndex.documentIds).includes(`${base}${prefix}experiments/${chapter.id}/#article-title`)) errors.add(`Chapter title missing from ${locale} search: ${chapter.id}`)
  }
}
const titles = new Map()
const descriptions = new Map()
const sitemap = load(await readFile(path.join(dist, 'sitemap.xml'), 'utf8'), { xml: true })
const locations = sitemap('url > loc').map((_, element) => sitemap(element).text()).get()
const canonicalUrls = new Set()
let links = 0
for (const [file, $] of documents) {
  const relative = path.relative(dist, file)
  if (!$('title').text()) errors.add(`${relative}: no title`)
  if (relative !== '404.html') {
    const normalized = relative.replace(/(^|\/)index\.html$/, '$1')
    const canonical = siteUrl + normalized
    canonicalUrls.add(canonical)
    if ($('link[rel="canonical"]').length !== 1 || $('link[rel="canonical"]').attr('href') !== canonical) errors.add(`${relative}: incorrect canonical`)
    const description = $('meta[name="description"]').attr('content')
    if (!description) errors.add(`${relative}: missing description`)
    for (const [value, registry, label] of [[$('head > title').text(), titles, 'title'], [description, descriptions, 'description']]) {
      if (registry.has(value)) errors.add(`${relative}: duplicate ${label} with ${registry.get(value)}`)
      registry.set(value, relative)
    }
    const en = relative.startsWith('en/')
    if ($('html').attr('lang') !== (en ? 'en' : 'zh-CN')) errors.add(`${relative}: incorrect html language`)
    const original = normalized.replace(/^en\//, '')
    for (const [language, expected] of [['zh-CN', siteUrl + original], ['en', siteUrl + 'en/' + original], ['x-default', siteUrl + original]]) {
      if ($(`link[rel="alternate"][hreflang="${language}"]`).attr('href') !== expected) errors.add(`${relative}: invalid ${language} alternate`)
      const local = expected.slice(siteUrl.length)
      const target = path.join(dist, local, local.endsWith('/') || !local ? 'index.html' : '')
      if (!documents.has(target)) errors.add(`${relative}: alternate points to missing page: ${expected}`)
    }
    if (($('meta[name="robots"]').attr('content') || '').includes('noindex')) errors.add(`${relative}: indexable page has noindex`)
    if ($('meta[property="og:url"]').attr('content') !== canonical) errors.add(`${relative}: incorrect Open Graph URL`)
    for (const property of ['og:title', 'og:description', 'og:image', 'og:locale']) if (!$(`meta[property="${property}"]`).attr('content')) errors.add(`${relative}: missing ${property}`)
    const schemas = $('script[type="application/ld+json"]').toArray().map(el => { try { return JSON.parse($(el).text()) } catch { errors.add(`${relative}: invalid JSON-LD`); return {} } })
    if (!schemas.length) errors.add(`${relative}: missing structured data`)
    for (const schema of schemas.filter(s => s['@type'] === 'BlogPosting')) {
      if (schema.url !== canonical || !schema.author?.name || !Number.isFinite(Date.parse(schema.dateModified)) || Date.parse(schema.dateModified) < Date.parse(schema.datePublished)) errors.add(`${relative}: invalid article metadata`)
    }
  } else if ($('meta[name="robots"]').attr('content') !== 'noindex') errors.add('404 must be noindex')

  if (relative !== '404.html' && $('h1').length !== 1) errors.add(`${relative}: expected one h1, got ${$('h1').length}`)
  if ($('mjx-merror, [data-mjx-error]').length) errors.add(`${relative}: invalid math`)
  for (const image of unrenderedMarkdownImages($)) errors.add(`${relative}: unrendered Markdown image: ${image}`)
  const current = new URL(base + relative, 'https://docs.local')
  for (const element of $('a[href], img[src], script[src], link[href]').toArray()) {
    const raw = $(element).attr('href') || $(element).attr('src')
    if (!raw || /^(?:https?:|mailto:|tel:|data:|javascript:|\/\/)/i.test(raw)) continue
    const url = new URL(raw, current)
    if (!url.pathname.startsWith(base)) { errors.add(`${relative}: URL escapes Pages base: ${raw}`); continue }
    let target = path.join(dist, decodeURIComponent(url.pathname.slice(base.length)))
    if ((await stat(target).catch(() => null))?.isDirectory()) target = path.join(target, 'index.html')
    if (!await stat(target).catch(() => null)) { errors.add(`${relative}: missing target: ${raw}`); continue }
    if (url.hash && documents.has(target)) {
      const fragment = decodeURIComponent(url.hash.slice(1))
      const targetDocument = documents.get(target)
      if (fragment && !targetDocument('[id]').toArray().some(el => targetDocument(el).attr('id') === fragment)) errors.add(`${relative}: missing anchor: ${raw}`)
    }
    links++
  }
}
// The downloadable/copyable Markdown bypasses VitePress's HTML URL rewriting.
// Audit it independently so missing Pages prefixes cannot hide behind valid HTML.
let markdownPages = 0
let markdownLinks = 0
for (const file of documents.keys()) {
  const relative = path.relative(dist, file)
  if (relative === '404.html') continue
  const markdownPath = relative.replace(/\.html$/, '.md')
  const markdown = await readFile(path.join(dist, markdownPath), 'utf8').catch(() => null)
  if (markdown === null) { errors.add(`${markdownPath}: missing Markdown export`); continue }
  markdownPages++
  for (const raw of collectMarkdownUrls(markdown)) {
    if (!raw || /^(?:mailto:|tel:|data:|javascript:)/i.test(raw)) continue
    if (!/^(?:[a-z]+:|\/\/|#)/i.test(raw)) errors.add(`${markdownPath}: non-portable Markdown URL: ${raw}`)
    const url = new URL(raw, siteUrl + relative)
    if (url.origin !== new URL(siteUrl).origin) continue
    if (!url.pathname.startsWith(base)) { errors.add(`${markdownPath}: URL escapes Pages base: ${raw}`); continue }
    let target = path.join(dist, decodeURIComponent(url.pathname.slice(base.length)))
    if ((await stat(target).catch(() => null))?.isDirectory()) target = path.join(target, 'index.html')
    if (!await stat(target).catch(() => null)) { errors.add(`${markdownPath}: missing target: ${raw}`); continue }
    if (url.hash && documents.has(target)) {
      const fragment = decodeURIComponent(url.hash.slice(1))
      const targetDocument = documents.get(target)
      if (fragment && !targetDocument('[id]').toArray().some(el => targetDocument(el).attr('id') === fragment)) errors.add(`${markdownPath}: missing anchor: ${raw}`)
    }
    markdownLinks++
  }
}
if (new Set(locations).size !== locations.length) errors.add('Sitemap contains duplicate URLs')
for (const url of locations) if (!canonicalUrls.has(url)) errors.add(`Sitemap URL has no canonical page: ${url}`)
for (const url of canonicalUrls) if (!locations.includes(url)) errors.add(`Canonical page missing from sitemap: ${url}`)
if (errors.size) {
  console.error([...errors].join('\n'))
  process.exitCode = 1
} else console.log(`Verified ${documents.size} HTML pages, ${links} internal links/assets, ${markdownPages} Markdown exports with ${markdownLinks} internal references and ${searchSections} search sections, including both languages, ${chapters.length} chapters per language, canonical URLs, hreflang, unique metadata, structured data and sitemap coverage.`)
