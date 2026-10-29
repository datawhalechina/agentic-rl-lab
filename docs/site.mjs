export const siteOrigin = 'https://datawhalechina.github.io'
export const siteBase = '/agentic-rl-lab/'
export const siteUrl = siteOrigin + siteBase
export const author = { name: 'KMnO4-zx', url: 'https://github.com/KMnO4-zx' }
export const routePath = (relativePath) => relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html')
export const pageUrl = (relativePath) => siteUrl + routePath(relativePath)
export const localizedPath = (path, locale) => (locale === 'en' ? '/en' : '') + path
