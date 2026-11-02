import assert from 'node:assert/strict'
import test from 'node:test'
import { load } from 'cheerio'
import MarkdownIt from 'markdown-it'
import { unrenderedMarkdownImages } from './rendered-images.mjs'

const md = new MarkdownIt({ html: true })
const audit = source => unrenderedMarkdownImages(load(`<div class="vp-doc">${md.render(source)}</div>`))

test('detects the missing DAPO figure even though HTML has no img to validate', () => {
  const source = '![03 · Clip Higher and Token Mean](./images/03 · Clip-Higher and Token Mean.png)'
  assert.equal(load(md.render(source))('img').length, 0)
  assert.deepEqual(audit(source), [source])
})

test('accepts angle-bracketed and encoded image paths with spaces', () => {
  for (const source of [
    '![图示](<./images/03 · Clip-Higher and Token Mean.png>)',
    '![图示](./images/03%20%C2%B7%20Clip-Higher%20and%20Token%20Mean.png)',
    '![图示][figure]\n\n[figure]: <./images/03 · Clip-Higher and Token Mean.png>',
  ]) {
    assert.equal(load(md.render(source))('img').length, 1)
    assert.deepEqual(audit(source), [])
  }
})

test('ignores literal image examples in fenced, indented and inline code', () => {
  const example = '![](./images/a b.png)'
  const source = '```md\n' + example + '\n```\n\n    ' + example + '\n\nExample: `' + example + '`.'
  assert.deepEqual(audit(source), [])
})

test('detects image syntax split by inline emphasis and missing references', () => {
  assert.deepEqual(audit('![**diagram**](./images/a b.png)'), ['![diagram](./images/a b.png)'])
  assert.deepEqual(audit('![diagram][missing]'), ['![diagram][missing]'])
})

test('excludes scripts, styles, navigation and syntax split across code or paragraphs', () => {
  const $ = load('<nav>![](./outside.png)</nav><div class="vp-doc"><script>![](./script.png)</script><style>![](./style.png)</style><p>![]<code>example</code>(./inline.png)</p><p>![]</p><p>(./next.png)</p></div>')
  assert.deepEqual(unrenderedMarkdownImages($), [])
})
