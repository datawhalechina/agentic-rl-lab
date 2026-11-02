import { readFile, writeFile, mkdir, copyFile, rm, stat } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import MarkdownIt from 'markdown-it'
import GithubSlugger from 'github-slugger'
import { chapters, localizedChapters, localizedGroups, repoUrl, chapterLink, articleCovers } from '../../docs/chapters.mjs'
import { sourceDates } from '../../docs/.vitepress/seo.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
const content = path.join(root, 'docs/content')
const media = path.join(content, 'public/media')
const parser = new MarkdownIt({ html: true })
const routes = new Map(chapters.flatMap(c => [
  [c.source, chapterLink(c)], ...(c.start ? [[c.start, `${chapterLink(c)}run`]] : []),
]))
routes.set('README.md', '/guide/introduction')
routes.set('README_EN.md', '/guide/introduction')
const copied = new Set()
const manifest = JSON.parse(await readFile(path.join(root, 'docs/translations/manifest.json'), 'utf8'))

// Clean only generated chapter directories and copied media, never handwritten pages.
await rm(media, { recursive: true, force: true })
for (const locale of ['zh', 'en']) for (const c of chapters) {
  await rm(path.join(content, locale === 'en' ? 'en' : '', 'experiments', c.id), { recursive: true, force: true })
}

async function resolveUrl(raw, source, image = false, locale = 'zh') {
  let url = raw.replace(/&amp;/g, '&')
  const githubPrefix = `${repoUrl}/blob/main/`
  if (url.startsWith(githubPrefix)) url = '/' + url.slice(githubPrefix.length)
  if (/^(?:[a-z]+:|\/\/|#)/i.test(url)) return raw
  const hashAt = url.indexOf('#')
  const hash = hashAt < 0 ? '' : url.slice(hashAt)
  const local = decodeURIComponent((hashAt < 0 ? url : url.slice(0, hashAt)).split('?')[0])
  const relative = path.posix.normalize(local.startsWith('/') ? local.slice(1) : path.posix.join(path.posix.dirname(source), local))
  if (relative.startsWith('../')) throw new Error(`Path escapes repository: ${source}: ${raw}`)
  if (routes.has(relative)) return (locale === 'en' ? '/en' : '') + routes.get(relative) + hash
  const absolute = path.join(root, relative)
  const info = await stat(absolute).catch(() => null)
  if (!info) throw new Error(`Missing local target: ${source}: ${raw} -> ${relative}`)
  if (image || /\.(png|jpe?g|gif|svg|webp|avif)$/i.test(relative)) {
    if (!copied.has(relative)) {
      await mkdir(path.dirname(path.join(media, relative)), { recursive: true })
      await copyFile(absolute, path.join(media, relative))
      copied.add(relative)
    }
    return '/media/' + relative.split('/').map(encodeURIComponent).join('/') + hash
  }
  return `${repoUrl}/${info.isDirectory() ? 'tree' : 'blob'}/main/${relative.split('/').map(encodeURIComponent).join('/')}${hash}`
}

async function transform(body, source, locale) {
  body = body.replace(/```math\r?\n([\s\S]*?)```/g, (_, equation) => `$$\n${equation.trim()}\n$$`)
  body = body.replace(/<div\b[^>]*>[\s\S]*?<\/div>/gi, block => /(?:shields\.io|visitor-badge|komarev\.com)/.test(block) ? '' : block)
  const destinations = new Map()
  function visit(tokens) {
    for (const token of tokens) {
      if (token.type === 'image') destinations.set(token.attrGet('src'), true)
      if (token.type === 'link_open') destinations.set(token.attrGet('href'), false)
      if (token.children) visit(token.children)
    }
  }
  visit(parser.parse(body, {}))
  for (const match of body.matchAll(/\b(src|href)=["']([^"']+)["']/g)) destinations.set(match[2], match[1] === 'src')
  for (const [url, isImage] of destinations) {
    const resolved = await resolveUrl(url, source, isImage, locale)
    if (resolved === url) continue
    for (const original of new Set([url, decodeURI(url)])) {
      body = body.split(`](${original})`).join(`](${resolved})`)
      body = body.split(`](<${original}>)`).join(`](${resolved})`)
      body = body.split(`src="${original}"`).join(`src="${resolved}"`)
      body = body.split(`href="${original}"`).join(`href="${resolved}"`)
      body = body.split(`src='${original}'`).join(`src='${resolved}'`)
      body = body.split(`href='${original}'`).join(`href='${resolved}'`)
    }
  }
  // Empty Markdown image labels become useful alt text based on the actual figure filename.
  body = body.replace(/!\[\]\(([^)]+)\)/g, (_, url) => {
    const label = decodeURIComponent(url.split('/').at(-1).split('#')[0]).replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ')
    return `![${label}](${url})`
  })
  return body
}

function sharedHeadingAnchors(original, translated) {
  // Preserve the Chinese source anchors in both editions so deep links and language switches work.
  const headings = text => {
    const tokens = parser.parse(text, {})
    return tokens.flatMap((token, i) => token.type === 'heading_open' ? [{ line: token.map[0], text: tokens[i + 1].children.filter(t => ['text', 'code_inline'].includes(t.type)).map(t => t.content).join(''), level: token.tag }] : [])
  }
  const originals = headings(original)
  const targets = headings(translated)
  if (originals.length !== targets.length || originals.some((h, i) => h.level !== targets[i].level)) throw new Error('Translation heading structure differs from the source')
  const lines = translated.split('\n')
  const slugger = new GithubSlugger()
  for (let i = 0; i < originals.length; i++) lines[targets[i].line] += ` {#${slugger.slug(originals[i].text)}}`
  return lines.join('\n')
}

function validateTranslation(original, translated, file) {
  const fences = value => parser.parse(value, {}).filter(token => token.type === 'fence').map(token => ({ info: token.info, content: token.content }))
  if (JSON.stringify(fences(original)) !== JSON.stringify(fences(translated))) throw new Error(`Translated code or fenced equation differs from source: ${file}`)
  if (/ZXQ\d+ZXQ|<end_of_turn>/.test(translated)) throw new Error(`Unresolved translation placeholder: ${file}`)
}

let pages = 0
for (const locale of ['zh', 'en']) for (const chapter of localizedChapters(locale)) {
  for (const isRun of [false, true]) {
    const source = isRun ? chapter.start : chapter.source
    if (!source) continue
    const original = await readFile(path.join(root, source), 'utf8')
    const translationSource = `docs/translations/en/${chapter.id}${isRun ? '.run' : ''}.md`
    let body = original
    if (locale === 'en') {
      if (manifest[translationSource]?.sourceSha256 !== createHash('sha256').update(original).digest('hex')) throw new Error(`Stale English translation: ${source}. Update the translation and manifest together.`)
      body = await readFile(path.join(root, translationSource), 'utf8')
      validateTranslation(original, body, translationSource)
    }
    body = body.replace(/^# .+\r?\n/, '')
    if (locale === 'en') {
      try { body = sharedHeadingAnchors(original.replace(/^# .+\r?\n/, ''), body) }
      catch (error) { throw new Error(`${translationSource}: ${error.message}`) }
    }
    const cover = !isRun && articleCovers[chapter.id]
    if (cover) {
      body = body.replace(/!\[[^\]]*\]\(([^)]+)\)/g, (match, url) => decodeURI(url) === cover ? '' : match)
      body = body.replace(/<div\b[^>]*>\s*<img\b[^>]*>\s*<\/div>/gi, block => block.includes(`src="${cover}"`) ? '' : block)
    }
    body = await transform(body, source, locale)
    const dates = sourceDates(source)
    const translationDate = locale === 'en' ? manifest[translationSource].translatedAt + 'T00:00:00+08:00' : undefined
    const frontmatter = {
      title: isRun ? `${chapter.title} ${locale === 'en' ? 'Quick Start: Setup and Training Commands' : '快速启动：环境、配置与运行命令'}` : chapter.seoTitle,
      description: isRun ? chapter.runDescription : chapter.description,
      chapter: chapter.id, source, ...(locale === 'en' ? { translationSource } : {}), isRun,
      ...(chapter.author ? { author: chapter.author } : {}),
      sourceDates: locale === 'en' ? { published: translationDate, modified: translationDate } : dates,
      readingTime: Math.max(1, Math.ceil(locale === 'en' ? body.replace(/```[\s\S]*?```/g, '').split(/\s+/).length / 220 : body.replace(/```[\s\S]*?```/g, '').length / 450)),
      group: localizedGroups(locale).find(g => g.id === chapter.group).label,
    }
    const output = path.join(content, locale === 'en' ? 'en' : '', 'experiments', chapter.id, isRun ? 'run.md' : 'index.md')
    await mkdir(path.dirname(output), { recursive: true })
    await writeFile(output, `---\n${Object.entries(frontmatter).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n')}\n---\n\n${body.trim()}\n`)
    pages++
  }
}
console.log(`Prepared ${pages} bilingual pages from ${chapters.length} chapters; copied ${copied.size} referenced images. Original research notes unchanged.`)
