# Git Worktrees Best Practices

**Last Updated**: 2025-10-27
**Status**: Required reading for parallel feature development

## Overview

Git worktrees allow multiple working directories from a single repository, enabling parallel development on different branches. However, they require specific setup steps to ensure pre-push hooks and development tools work correctly.

## Critical Setup Steps

### 1. Creating a Worktree

```bash
cd /home/gerald/git/full-duplex-voice-chat

# Create worktree for new feature
git worktree add ../full-duplex-voice-chat-feature -b feature/feature-name

# Verify creation
git worktree list
```

### 2. **CRITICAL**: Sync Development Dependencies

**⚠️ ALWAYS run this immediately after creating a worktree:**

```bash
cd ../full-duplex-voice-chat-feature

# Install ALL dependencies including dev tools
uv sync --all-extras

# Verify critical tools are available
uv run mypy --version
uv run ruff --version
uv run pytest --version
```

**Why This Is Critical:**
- Git worktrees share `.git` but have **independent Python environments**
- Pre-push hooks rely on dev dependencies (mypy, ruff, pytest)
- Without sync, hooks fail silently or report false positives
- CI will catch errors that local hooks miss

### 3. Generate Proto Files (If Applicable)

```bash
# If project uses protobuf
uv run just gen-proto
```

### 4. Verify Pre-Push Hook Works

**Test the hook manually before making commits:**

```bash
# Test hook
.git/hooks/pre-push origin refs/heads/feature/feature-name

# Expected output:
# 🔍 Running pre-push validation...
# ✓ Linting passed
# ✓ Type checking passed
# ✓ Unit tests passed
# ✓ Pre-push validation passed!
```

**If hook fails with "command not found":**
- You didn't run `uv sync --all-extras`
- Go back to step 2

## Common Pitfalls and Solutions

### Pitfall 1: "mypy: command not found"

**Problem**: Dev dependencies not installed in worktree environment

**Solution**:
```bash
cd /path/to/worktree
uv sync --all-extras
```

### Pitfall 2: Type Stubs Missing

**Problem**: Mypy can't find type information for third-party packages

**Solution**:
```bash
# Install type stubs (included in dev dependencies)
uv sync --all-extras

# Verify stubs installed
uv pip list | grep types-
```

Common type stubs in this project:
- `types-redis`
- `types-pyyaml`
- `types-requests`

### Pitfall 3: Using `--no-verify` to Bypass Hooks

**Problem**: Pushing with `--no-verify` bypasses quality checks

**❌ NEVER DO THIS**:
```bash
git push --no-verify  # Bypasses ALL checks
```

**✅ DO THIS INSTEAD**:
```bash
# Fix the underlying issue
uv sync --all-extras
uv run just ci

# Then push normally
git push
```

**Exceptions**: Only use `--no-verify` for:
- Emergency hotfixes (with explicit team approval)
- Retrospective PRs where code already passed CI in main
- NEVER for new feature development

### Pitfall 4: Tests Pass Locally, Fail in CI

**Problem**: Different environments between worktree and CI

**Common Causes**:
1. Forgot `uv sync --all-extras`
2. Used `--no-verify` to bypass checks
3. Stale Python cache in worktree

**Solution**:
```bash
# Clean environment
uv sync --all-extras
uv cache clean

# Run full CI suite locally
uv run just ci

# Verify ALL checks pass
uv run just lint
uv run just typecheck
uv run just test
```

## Worktree Workflow Checklist

Use this checklist for every worktree:

- [ ] Create worktree: `git worktree add ../project-feature -b feature/name`
- [ ] Navigate to worktree: `cd ../project-feature`
- [ ] **Sync all dependencies**: `uv sync --all-extras`
- [ ] Generate proto files (if needed): `uv run just gen-proto`
- [ ] Test pre-push hook: `.git/hooks/pre-push origin refs/heads/feature/name`
- [ ] Verify tools available: `uv run mypy --version`
- [ ] Make changes and commit
- [ ] Run full CI locally: `uv run just ci`
- [ ] Push (hook will run automatically)
- [ ] Create PR
- [ ] Clean up after merge: `git worktree remove ../project-feature`

## Pre-Push Hook Verification

### What the Hook Checks

1. **Linting** (`ruff check`)
   - Code style
   - Import sorting
   - Line length
   - Common errors

2. **Type Checking** (`mypy`)
   - Type annotations
   - Type consistency
   - Missing imports
   - Generic type parameters

3. **Unit Tests** (`pytest -m "not infrastructure"`)
   - Core logic tests
   - Mock-based tests
   - Fast tests only

### Manual Hook Testing

```bash
# Run hook manually
.git/hooks/pre-push origin refs/heads/feature/branch-name

# Check exit code
echo $?  # Should be 0 for success

# If hook fails, fix issues before pushing
uv run just ci
```

### Hook Failure Troubleshooting

**Error: "mypy: command not found"**
```bash
uv sync --all-extras
```

**Error: "Cannot find implementation or library stub"**
```bash
# Missing type stubs
uv sync --all-extras

# Check stubs installed
uv pip list | grep types-
```

**Error: Tests failed**
```bash
# Run tests to see failures
uv run just test

# Fix failures, then push
```

## Worktree Management

### List All Worktrees

```bash
git worktree list

# Output:
# /path/to/main        abc1234 [main]
# /path/to/feature1    def5678 [feature/m9-routing]
# /path/to/feature2    ghi9012 [feature/m12-docker]
```

### Remove Worktree After Merge

```bash
# After PR merges to main
cd /home/gerald/git/full-duplex-voice-chat

# Remove worktree
git worktree remove ../full-duplex-voice-chat-feature

# Delete local branch
git branch -D feature/feature-name

# Update main
git checkout main
git pull origin main
```

### Force Remove (If Needed)

```bash
# If worktree has uncommitted changes
git worktree remove --force ../full-duplex-voice-chat-feature
```

## Parallel Development Pattern

### Scenario: Two Features Simultaneously

**Setup (Coordinator Agent)**:
```bash
cd /home/gerald/git/full-duplex-voice-chat
git checkout main
git pull origin main

# Create worktrees
git worktree add ../project-feature1 -b feature/feature1
git worktree add ../project-feature2 -b feature/feature2

# Setup BOTH worktrees
cd ../project-feature1 && uv sync --all-extras && cd -
cd ../project-feature2 && uv sync --all-extras && cd -
```

**Parallel Development**:
- Team A works in `../project-feature1`
- Team B works in `../project-feature2`
- No conflicts (different branches)
- Independent CI checks
- Separate PRs

**Cleanup After Merges**:
```bash
cd /home/gerald/git/full-duplex-voice-chat
git worktree remove ../project-feature1
git worktree remove ../project-feature2
git branch -D feature/feature1 feature/feature2
git checkout main && git pull
```

## Integration with Claude Code Agents

### Agent Instructions

When coordinating parallel feature development:

1. **Create worktrees for each team**
2. **Document worktree paths in task descriptions**
3. **Explicitly instruct agents to run `uv sync --all-extras`**
4. **Test hooks before starting work**
5. **Never allow `--no-verify` in agent prompts**

**Example Agent Prompt**:
```markdown
You are working in: /home/gerald/git/full-duplex-voice-chat-feature
Branch: feature/feature-name

**SETUP REQUIRED (Run These First)**:
```bash
cd /home/gerald/git/full-duplex-voice-chat-feature
uv sync --all-extras
uv run just gen-proto
.git/hooks/pre-push origin refs/heads/feature/feature-name
```

After setup verification, implement the feature...
```

### Agent Workflow Validation

Agents should verify setup before starting:

```bash
# 1. Verify uv environment
uv run python --version

# 2. Verify dev tools
uv run mypy --version
uv run ruff --version
uv run pytest --version

# 3. Verify proto files generated (if applicable)
ls -la src/rpc/generated/

# 4. Test pre-push hook
.git/hooks/pre-push origin refs/heads/$(git branch --show-current)

# 5. Proceed with implementation
```

## References

- **Git Worktree Docs**: https://git-scm.com/docs/git-worktree
- **uv Sync**: `.claude/modules/development.md#environment-setup`
- **Pre-Push Hook**: `.git/hooks/pre-push`
- **Testing Strategy**: `.claude/modules/testing.md`
- **Development Workflow**: `.claude/modules/development.md`

## Lessons Learned

### October 2025: M9/M12 Parallel Implementation

**Issue**: Type check errors in CI that weren't caught locally

**Root Cause**:
- Created worktrees for parallel M9/M12 development
- Forgot to run `uv sync --all-extras` in worktrees
- Pre-push hooks failed silently (mypy not found)
- Used `--no-verify` to bypass, pushing broken code to CI

**Resolution**:
- Fixed type errors in CI
- Documented worktree setup requirements
- Updated agent coordination patterns
- Added this guide to prevent recurrence

**Key Takeaway**: **NEVER skip `uv sync --all-extras` in worktrees**

---

**Remember**: Worktrees are powerful for parallel development, but they require proper setup. Follow this guide religiously to avoid CI failures and maintain code quality.
