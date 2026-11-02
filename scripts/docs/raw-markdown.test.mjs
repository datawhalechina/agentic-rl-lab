import assert from 'node:assert/strict'
import test from 'node:test'
import { rewriteRawMarkdown, collectMarkdownUrls } from '../../docs/.vitepress/raw-markdown.mjs'
import { siteUrl } from '../../docs/site.mjs'

const pages = ['experiments/dapo/index.md', 'experiments/dapo/run.md', 'guide/learning-path.md', 'en/experiments/dapo/index.md']
const rewrite = value => rewriteRawMarkdown(value, 'experiments/dapo/index.md', pages)

test('exported links use the deployed base and HTML routes, preserving queries and anchors', () => {
  const source = '[run](/experiments/dapo/run?mode=short#setup) [relative](run.md) [guide](/guide/learning-path) [English](/en/experiments/dapo/) [base](/agentic-rl-lab/experiments/dapo/) [anchor](#result) [external](https://example.com/a)'
  const result = collectMarkdownUrls(rewrite(source))
  assert.deepEqual(result, [siteUrl + 'experiments/dapo/run.html?mode=short#setup', siteUrl + 'experiments/dapo/run.html', siteUrl + 'guide/learning-path.html', siteUrl + 'en/experiments/dapo/', siteUrl + 'experiments/dapo/', '#result', 'https://example.com/a'])
})

test('images, titled links, HTML attributes and reference definitions stay valid Markdown', () => {
  const source = `![Figure](</media/图表 with space.png> "Caption")
[guide](/guide/learning-path 'Read more')
[quick start][run]

[run]: /experiments/dapo/run "Start"

<img src="/media/a.png" alt="Figure">
<a href='/guide/learning-path'>Guide</a>`
  const output = rewrite(source)
  assert.match(output, /"Caption"/)
  assert.match(output, /'Read more'/)
  assert.match(output, /"Start"/)
  assert.deepEqual(collectMarkdownUrls(output), [siteUrl + 'media/%E5%9B%BE%E8%A1%A8%20with%20space.png', siteUrl + 'guide/learning-path.html', siteUrl + 'experiments/dapo/run.html', siteUrl + 'media/a.png', siteUrl + 'guide/learning-path.html'])
})

test('frontmatter and fenced, indented and inline examples remain byte-for-byte unchanged', () => {
  const frontmatter = '---\ntitle: "[Example](/guide/learning-path)"\n---\n'
  const examples = '`[Example](/guide/learning-path)`\n\n```markdown\n[Example](/guide/learning-path)\n<img src="/media/a.png">\n```\n\n    [Example](/guide/learning-path)\n'
  const actual = '[Example](/guide/learning-path)'
  assert.equal(rewrite(frontmatter + examples + '\n' + actual), frontmatter + examples + '\n[Example](' + siteUrl + 'guide/learning-path.html)')
  assert.deepEqual(collectMarkdownUrls(frontmatter + examples + '\n' + actual), ['/guide/learning-path'])
})

test('URL collection ignores code examples while retaining parsed reference and HTML media', () => {
  const source = '[ref][target] ![figure](/media/a.png) `<img src="/media/example.png">`\n\n[target]: /guide/learning-path\n\n<img src="/media/html.png">'
  assert.deepEqual(collectMarkdownUrls(source), ['/guide/learning-path', '/media/a.png', '/media/html.png'])
})

test('escaped parentheses in image filenames remain part of the exported URL', () => {
  const source = String.raw`![figure](./figure\).png)`
  assert.deepEqual(collectMarkdownUrls(rewrite(source)), [siteUrl + 'experiments/dapo/figure%29.png'])
})
