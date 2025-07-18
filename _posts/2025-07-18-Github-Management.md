---
layout: post
title: GitHub Website Management Guide
date: 2025-07-18 14:24:00
description: A comprehensive guide to managing GitHub repositories for personal website deployment, including code commits, branch management, and deployment workflows
tags: github website deployment automation
categories: web-development
---

This blog post documents how to manage GitHub repositories for personal website deployment. My GitHub username is `Ziya-Jiang`, and this guide will cover the complete workflow from local development to GitHub Pages deployment.

## Part 1: Committing Updated Repository to Main Branch

### 1. Initialize Local Repository

First, ensure your local project is initialized as a Git repository:

```bash
# Initialize Git repository if not already done
git init

# Add remote repository (if not already added)
git remote add origin https://github.com/Ziya-Jiang/Ziya-Jiang.github.io.git
```

### 2. Check Current Status

Before committing, check the current working status:

```bash
# Check current branch
git branch

# Check file status
git status

# View modified files
git diff
```

### 3. Add Files to Staging Area

Add all modified files to the Git staging area:

```bash
# Add all files
git add .

# Or add specific files
git add _posts/2025-07-18-Github-Management.md
git add _config.yml
```

### 4. Commit Changes

Commit changes with meaningful commit messages:

```bash
# Commit changes
git commit -m "feat: add GitHub website management blog post

- Add comprehensive GitHub repository management guide
- Include code commit and branch management workflow
- Update website configuration and content"
```

### 5. Push to Main Branch

Push local changes to the GitHub main branch:

```bash
# Push to main branch
git push origin main

# If first time pushing, may need to set upstream branch
git push -u origin main
```

### 6. Verify Push Results

After pushing, verify the results:

```bash
# Check remote branch status
git remote -v

# View commit history
git log --oneline -5
```

### 7. Troubleshooting Common Issues

#### If you encounter push conflicts:

```bash
# First pull remote changes
git pull origin main

# Resolve conflicts and recommit
git add .
git commit -m "resolve: fix merge conflicts"
git push origin main
```

#### If force push is needed (use with caution):

```bash
# Force push (use only when necessary)
git push --force origin main
```

### 8. Automation Script

To simplify the process, create an automation script:

```bash
#!/bin/bash
# deploy.sh

echo "Starting deployment process..."

# Add all changes
git add .

# Commit changes
git commit -m "update: $(date '+%Y-%m-%d %H:%M:%S') auto update"

# Push to main branch
git push origin main

echo "Deployment completed!"
```

Usage:
```bash
chmod +x deploy.sh
./deploy.sh
```

### 9. Best Practices

1. **Regular commits**: Don't accumulate too many changes before committing
2. **Meaningful commit messages**: Use clear commit messages to describe changes
3. **Branch management**: Consider using feature branches for important changes
4. **Backup**: Regularly backup important files
5. **Testing**: Test website functionality locally before pushing

### 10. Next Steps

In the next part, we will cover:
- GitHub Pages configuration and deployment
- Custom domain setup
- Automated deployment workflows
- Performance optimization and monitoring

---
 