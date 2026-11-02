// A malformed Markdown image remains visible prose and never becomes an <img>.
// Checking only resource attributes would therefore silently miss it.
export function unrenderedMarkdownImages($) {
  const body = $('.vp-doc').clone()
  // Keep literal Markdown examples and non-content markup out of the audit.
  body.find('pre, code, script, style').replaceWith('\n')
  body.find('br').replaceWith('\n')
  body.find('p, li, td, th, blockquote, figcaption, h1, h2, h3, h4, h5, h6').append('\n')
  const syntax = /!\[[^\]\r\n]*\](?:\([^\r\n]*?\)|\[[^\]\r\n]*\])/g
  return [...new Set(body.text().match(syntax) || [])]
}
