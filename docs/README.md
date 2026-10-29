# Agentic RL Lab 在线实验手册

网站使用 VitePress，主题与首页位于 `docs/.vitepress/`。原有实验目录中的 Markdown 是正文的唯一来源。构建前，`scripts/docs/prepare.mjs` 读取 `docs/chapters.mjs`，生成章节页面、转换站内链接并复制引用到的图片；不会改写原文。

## 本地预览

需要 Node.js 22 或以上；GitHub Actions 使用 Node.js 24。

```bash
npm ci
npm run docs:dev
```

开发入口：`http://localhost:5173/agentic-rl-lab/`。修改章节原文后重新运行 `npm run docs:prepare`；首页、主题和手写指南支持开发服务器热更新。

```bash
npm run docs:build
npm run docs:check
npm run docs:preview
```

`docs:check` 检查构建结果中的全部站内链接、图片、锚点、章节、页面标题和数学错误。Pages 项目路径已经固定为 `/agentic-rl-lab/`，深层页面使用 `.html`，不依赖托管平台重写 URL。

## 内容结构

- `docs/chapters.mjs`：章节、分组、简介与原始 Markdown 路径。
- `docs/content/guide/`：网站独有的介绍、学习路线、环境准备、算法对照、PyTRIO 教程与 Tinker 工作流对照；英文对应页位于 `docs/content/en/guide/`。
- `docs/content/experiments/index.md`：可筛选的实验目录。
- `docs/content/experiments/*/`：自动生成的文章和快速启动，不提交。
- `docs/content/public/media/`：自动复制的正文配图，不提交。
- `docs/.vitepress/theme/`：首页、阅读页、主题与响应式样式。

新增文章时，在章节注册表增加一项即可。保留原文中的论文、实现与结果说明；网页仅替换页头、移除注册表明确列出的宣传封面、访问统计及 badge 区域，并转换链接、GitHub 数学块和图片路径。算法图与实验结果图全部保留，原始 Markdown 和图片不变。正文编辑入口直接指向 `main` 分支中的原始文件。

## GitHub Pages

当前隔离开发分支为 `doc`。`.github/workflows/docs.yml` 在推送该分支后构建、校验并部署；PR 仅构建和校验。部署前需在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**，并允许 `doc` 分支使用 `github-pages` environment。

部署后的地址为 `https://datawhalechina.github.io/agentic-rl-lab/`。

后续 `main` 增加或修改正文后，先在 `doc` 合并最新 `main`，验证构建再推送。建议网站稳定后把网站源码合回 `main`，同时将 workflow 的 `push.branches` 和部署条件改为 `main`、手写网页的编辑链接改为 `edit/main`。这样正文变更会直接触发部署，避免长期维护两条内容分支。

依赖目录、生成正文、复制图片、构建产物及浏览器验证截图均在根目录 `.gitignore` 中排除；`package-lock.json` 必须保留，用于可复现构建。

## 字体

DM Sans 和 IBM Plex Mono 通过 Fontsource 打包为本地静态资源，不依赖 Google Fonts 在线请求。两款字体的 OFL 许可及版权声明位于 `docs/content/public/licenses/`，随站点一同发布；中文采用系统字体。

## 中英文内容与 SEO

中文保留原地址，英文放在 `/en/`。两种语言各有独立站内搜索、导航、目录和完整正文，文章间可以切换语言。`docs/metadata.mjs` 维护面向搜索的标题、文章/快速启动的独立简介和英文目录文案。

英文正文位于 `docs/translations/en/`，按照对应中文文件的相对路径解析图片和代码链接。`docs/translations/manifest.json` 记录中文来源的 SHA-256。更新原文后，需要检查并同步英文译文，再更新对应 hash；构建会拒绝未同步的内容。代码块、数学表达式和原始结果图沿用原文，正文翻译可以单独编辑。英文标题保留对应中文章节的锚点，以支持深链接与语言切换。

`docs/.vitepress/seo.mjs` 为页面生成 canonical、双向 hreflang、Open Graph、Twitter 卡片和文章/面包屑结构化数据；这些元数据也会随客户端导航更新。原文日期从 Git 历史读取，Actions 使用完整历史，避免把每次部署都当作文章更新。英文首次上线日期由翻译清单记录。

分享图片为 `docs/content/public/social-card.png`，可编辑源文件为同目录的 SVG。Google 与 Bing 验证标签由各自站长后台提供，保留在配置中以维持验证状态。首次收录与后续效果复查见 [SEO review record](./seo-review.md)。
