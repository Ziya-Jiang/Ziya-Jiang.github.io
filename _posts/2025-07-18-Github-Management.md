---
layout: post
title: GitHub网站管理指南
date: 2025-07-18 14:24:00
description: 记录如何管理个人网站部署的GitHub仓库，包括代码提交、分支管理、部署流程等
tags: github website deployment
categories: web-development
---

# GitHub网站管理指南

本博客主要记录如何管理个人网站部署的GitHub仓库。我的GitHub用户名是 `Ziya-Jiang`，本文将详细介绍从本地开发到GitHub Pages部署的完整流程。

## 第一部分：将更新后的仓库提交到main分支

### 1. 初始化本地仓库

首先，确保您的本地项目已经初始化为Git仓库：

```bash
# 如果还没有初始化Git仓库
git init

# 添加远程仓库（如果还没有添加）
git remote add origin https://github.com/Ziya-Jiang/Ziya-Jiang.github.io.git
```

### 2. 检查当前状态

在提交之前，先检查当前的工作状态：

```bash
# 查看当前分支
git branch

# 查看文件状态
git status

# 查看修改的文件
git diff
```

### 3. 添加文件到暂存区

将所有修改的文件添加到Git暂存区：

```bash
# 添加所有文件
git add .

# 或者添加特定文件
git add _posts/2025-07-18-Github网站管理.md
git add _config.yml
```

### 4. 提交更改

使用有意义的提交信息提交更改：

```bash
# 提交更改
git commit -m "feat: 添加GitHub网站管理博客文章

- 新增GitHub仓库管理指南
- 包含代码提交和分支管理流程
- 更新网站配置和内容"
```

### 5. 推送到main分支

将本地更改推送到GitHub的main分支：

```bash
# 推送到main分支
git push origin main

# 如果是第一次推送，可能需要设置上游分支
git push -u origin main
```

### 6. 验证推送结果

推送完成后，可以通过以下方式验证：

```bash
# 查看远程分支状态
git remote -v

# 查看提交历史
git log --oneline -5
```

### 7. 常见问题解决

#### 如果遇到推送冲突：

```bash
# 先拉取远程更改
git pull origin main

# 解决冲突后重新提交
git add .
git commit -m "resolve: 解决合并冲突"
git push origin main
```

#### 如果需要强制推送（谨慎使用）：

```bash
# 强制推送（仅在必要时使用）
git push --force origin main
```

### 8. 自动化脚本

为了简化流程，可以创建一个自动化脚本：

```bash
#!/bin/bash
# deploy.sh

echo "开始部署流程..."

# 添加所有更改
git add .

# 提交更改
git commit -m "update: $(date '+%Y-%m-%d %H:%M:%S') 自动更新"

# 推送到main分支
git push origin main

echo "部署完成！"
```

使用方法：
```bash
chmod +x deploy.sh
./deploy.sh
```

### 9. 最佳实践

1. **定期提交**：不要积累太多更改再提交
2. **有意义的提交信息**：使用清晰的提交信息描述更改内容
3. **分支管理**：对于重要更改，考虑使用功能分支
4. **备份**：定期备份重要文件
5. **测试**：推送前在本地测试网站功能

---
