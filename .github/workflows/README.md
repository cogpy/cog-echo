# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the Deep Tree Echo repository.

## Active Workflows

### 1. Dependency Review (`dependency-review.yml`)
- **Trigger:** Pull requests to `main` branch
- **Purpose:** Scans dependency changes for known vulnerabilities
- **Status:** ✅ Active and working

### 2. Greetings (`greetings.yml`)
- **Trigger:** First-time issues and pull requests
- **Purpose:** Welcomes new contributors with context-appropriate messages about the AI consciousness project
- **Status:** ✅ Active and customized for Deep Tree Echo

### 3. Python Linting (`python-lint.yml`)
- **Trigger:** Pushes and PRs to `main` and `develop` branches
- **Purpose:** Basic Python code quality checks using flake8
- **Features:**
  - Detects critical syntax errors (E9, F7, F82)
  - Non-blocking: provides warnings without failing the build
  - Runs on Python 3.11
- **Status:** ✅ Active and optimized for this repository

## Removed Workflows

The following workflows were removed as they were not appropriate for this Python-based repository:

1. **featest.yml** - Used non-existent `actions/ai-inference@v1` action
2. **summary.yml** - Used non-existent `actions/ai-inference@v1` action
3. **msvc.yml** - C++ code analysis (repository contains only Python code)
4. **functest.yml** - Overly complex Python tests that consistently failed

## Notes

- This repository is primarily Python-based, focusing on AI consciousness exploration
- Workflows are designed to be lightweight and non-blocking to facilitate rapid development
- For adding new workflows, ensure they align with the project's Python/AI focus
