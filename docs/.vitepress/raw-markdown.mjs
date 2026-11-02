import MarkdownIt from 'markdown-it'
import { siteUrl, siteBase, routePath } from '../site.mjs'

const parser = new MarkdownIt({ html: true })

// Keep source formatting (and examples) intact while rewriting actual link
// destinations. Markdown-it handles inline code and escaped delimiters for us.
function editableRanges(markdown) {
  const lines = markdown.split('\n')
  const offsets = [0]
  for (const line of lines) offsets.push(offsets.at(-1) + line.length + 1)
  const excluded = parser.parse(markdown, {}).filter(token => ['fence', 'code_block'].includes(token.type)).map(token => [offsets[token.map[0]], offsets[token.map[1]]])
  const frontmatter = markdown.match(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/)
  if (frontmatter) excluded.push([0, frontmatter[0].length])
  const ranges = []
  let start = 0
  for (const [from, to] of excluded.sort((a, b) => a[0] - b[0])) {
    if (from > start) ranges.push([start, from])
    start = Math.max(start, to)
  }
  if (start < markdown.length) ranges.push([start, markdown.length])
  return ranges
}

function destinationEdits(source, resolve) {
  const edits = new Map()
  const record = (start, end, url, angled = false, html = false) => {
    let next = resolve(url)
    // parseLinkDestination unescapes \\( and \\); bare destinations must retain their boundaries.
    if (next !== url && !angled && !html) next = next.replace(/\(/g, '%28').replace(/\)/g, '%29')
    if (next !== url) edits.set(start, { start, end, value: angled ? `<${next}>` : next })
  }
  const inline = new MarkdownIt({ html: true })
  inline.inline.ruler.before('link', 'export_destinations', (state, silent) => {
    if (silent || state.src !== source) return false
    const pos = state.pos
    const image = state.src.startsWith('![', pos)
    const labelStart = image ? pos + 1 : pos
    if (state.src[labelStart] === '[') {
      const labelEnd = state.md.helpers.parseLinkLabel(state, labelStart, false)
      if (labelEnd >= 0 && state.src[labelEnd + 1] === '(') {
        let start = labelEnd + 2
        while (/\s/.test(state.src[start] || '') && start < state.posMax) start++
        const result = state.md.helpers.parseLinkDestination(state.src, start, state.posMax)
        if (result.ok) {
          let end = result.pos
          while (/\s/.test(state.src[end] || '') && end < state.posMax) end++
          if (end > result.pos) {
            const title = state.md.helpers.parseLinkTitle(state.src, end, state.posMax)
            if (title.ok) { end = title.pos; while (/\s/.test(state.src[end] || '') && end < state.posMax) end++ }
          }
          if (state.src[end] === ')') record(start, result.pos, result.str, state.src[start] === '<')
        }
      }
    }
    if (state.src[pos] === '<') {
      const tag = state.src.slice(pos).match(/^<\/?[A-Za-z][^>]*>/)?.[0]
      if (tag) for (const match of tag.matchAll(/\b(?:src|href)=(['"])(.*?)\1/g)) {
        const start = pos + match.index + match[0].indexOf(match[1]) + 1
        record(start, start + match[2].length, match[2], false, true)
      }
    }
    return false
  })
  inline.inline.parse(source, inline, {}, [])
  // Reference definitions are consumed by the block parser, not the inline one.
  for (const match of source.matchAll(/^ {0,3}\[[^\]\n]+\]:[ \t]*/gm)) {
    const start = match.index + match[0].length
    const result = parser.helpers.parseLinkDestination(source, start, source.length)
    if (result.ok) record(start, result.pos, result.str, source[start] === '<')
  }
  return [...edits.values()].sort((a, b) => b.start - a.start)
}

export function rewriteRawMarkdown(markdown, relativePath, pagePaths) {
  const pages = new Set(pagePaths)
  const resolve = raw => {
    if (!raw || /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(raw)) return raw
    const url = new URL(raw.replace(/&amp;/g, '&'), siteUrl + relativePath)
    // Source Markdown uses root-relative paths; VitePress adds base only when
    // rendering HTML. Explicitly include it in exported/copied Markdown.
    if (raw.startsWith('/') && !url.pathname.startsWith(siteBase)) url.pathname = siteBase.slice(0, -1) + url.pathname
    if (!url.pathname.startsWith(siteBase)) return raw
    const local = decodeURIComponent(url.pathname.slice(siteBase.length))
    const candidates = [local, local + '.md', local.replace(/\/$/, '') + '/index.md']
    const page = candidates.find(candidate => pages.has(candidate))
    if (page) url.pathname = siteBase + routePath(page)
    return url.href
  }
  let output = markdown
  for (const [start, end] of editableRanges(markdown).reverse()) {
    let source = markdown.slice(start, end)
    for (const edit of destinationEdits(source, resolve)) source = source.slice(0, edit.start) + edit.value + source.slice(edit.end)
    output = output.slice(0, start) + source + output.slice(end)
  }
  return output
}

export function collectMarkdownUrls(markdown) {
  const urls = []
  function visit(tokens) {
    for (const token of tokens) {
      if (token.type === 'image') urls.push(token.attrGet('src'))
      if (token.type === 'link_open') urls.push(token.attrGet('href'))
      if (['html_inline', 'html_block'].includes(token.type)) for (const match of token.content.matchAll(/\b(?:src|href)=(['"])(.*?)\1/g)) urls.push(match[2])
      if (token.children) visit(token.children)
    }
  }
  visit(parser.parse(markdown.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, ''), {}))
  return urls
}
