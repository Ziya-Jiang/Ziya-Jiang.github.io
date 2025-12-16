---
layout: post
title: Site Maintenance & Recent Updates
date: 2025-12-02 10:27:00-0400
inline: false
related_posts: True
---

This announcement records the recent major changes and long-term maintenance plan for this website.

---

### 1. 个人信息与 About 页面

- 更新了首页 `About` 页面个人简介，加入了我在哈尔滨成长、在机器人实验室长大的背景，以及关于“赋予机器人真正灵魂”和“正确的信仰”的想法。
- 将联系方式、地址统一为当前在深圳大学城北大校区的办公室信息。
- 简历 JSON (`assets/json/resume.json`) 中同步更新：
  - 个人简介（summary）
  - 地址、电话、邮箱
  - 教育经历（北大深研院 & 大连理工）
  - 奖项（国家奖学金、优秀毕业生、学习优秀奖等）
  - 语言能力（含 CET-4 / CET-6 分数）
  - 工程与资助项目（与项目页联动）。

---

### 2. Publications 与引用系统

- 清空了模板中的爱因斯坦示例文献，`_bibliography/papers.bib` 只保留/新增了我自己的 5 篇论文：
  - TAMBRIDGE（ArXiv，含封面预览图与 PDF）
  - Uncertainty-Driven 3D Gaussian Splatting for Robust Real-Time RGB-D SLAM（T-ASE under review）
  - ICRA 2025 物体位姿估计论文
  - AAAI-26 Debiased Multiplex Tokenizer
  - ACAIT 2023 Visual Odometry 架构论文
- 为每篇论文配置了：
  - `preview`：位于 `assets/img/publication_preview/` 的 GIF / JPEG 封面
  - `pdf`：位于 `assets/pdf/` 的 PDF 文件
  - 正确的 `year`、`booktitle`、作者列表和链接。

---

### 3. Projects 页面重构

- 用真实项目内容替换了 `_projects/1_project.md`–`4_project.md` 模板示例：
  1. **Text-Driven Simulation Scene Generation**（广东省旗舰项目）
  2. **Intelligent Dual-Arm Nursing Robot System**（国家重点研发计划）
  3. **Multi-Robot Distributed Perception**（国家自然科学基金项目）
  4. **Unmanned Supermarket Navigation**（深圳市稳定支持项目）
- 每个项目包括：
  - Overview / Role / Funding Source
  - 个人贡献（算法设计、系统架构、平台搭建等）
  - 使用的关键技术与工具。
- 第 3 个项目启用了 `related_publications: true`，并通过 `{% cite jiang2024tambridge jiang2025uncertainty %}` 关联了 TAMBRIDGE 和 Uncertainty SLAM 两篇论文，自动生成 References 区块。

---

### 4. Repositories 与社交信息

- `_data/repositories.yml` 中替换为我的 GitHub 信息：
  - `github_users: [Ziya-Jiang]`
  - `github_repos: [Ziya-Jiang/Ziya-Jiang.github.io]`
- 未来会逐步补充更多开源仓库（例如 SLAM / 具身智能相关工程），并在 Projects 与 Repositories 页面保持一致展示。

---

### 5. 维护计划与承诺

- 本网站将作为我在 **机器人、SLAM 与具身智能** 方向的长期主页，持续维护：
  - 新论文与项目进展会在 Publications / Projects / News 中同步更新；
  - 尽量保持 URL 与页面结构稳定，避免历史链接失效。
- 所有改动均通过 Git 版本管理并托管在 `Ziya-Jiang/Ziya-Jiang.github.io` 仓库中，保证内容可追溯、可备份、可迁移。

如你在浏览过程中发现任何错误（链接失效、排版问题等），欢迎通过 GitHub Issues 或邮件与我联系。
