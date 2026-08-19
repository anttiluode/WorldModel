# WorldModel

> **Let the model imagine the unseen world. Never let it forget which parts it imagined.**

`main` is intentionally a **navigation / landing branch** for now. The working experiments live on separate branches so old results and failure modes stay reproducible instead of being overwritten.

## Branch map

| Branch | What it contains |
|---|---|
| `agent/anchored-world-v0` | The anchored-world bookkeeping work: separate world belief from external support / provenance. |
| `agent/worldsplat-studio` | The first practical WorldSplat trainer/viewer. RGB-only CelebA produced the small rotatable, but geometrically wrong, hollow-face-style scene manifold. |
| `agent/worldsplat-vkitti2` | Virtual KITTI 2 version with real RGB + metric depth supervision. This is the completed 80k control run. |
| `agent/worldsplat-vkitti2-rayfix` | Current Phase-1 A/B branch. Fixes the ray/depth coordinate conflict and the asymmetric non-square anchor grid, while leaving the previous VKITTI2 run untouched. |

The current active experiment is usually the most recent branch above, but **do not assume newer means better**. Several branches exist specifically as controls.

---

# Getting the repository

## Option A — normal clone, then move between branches

```bash
git clone https://github.com/anttiluode/WorldModel.git
cd WorldModel

git fetch --all --prune
git branch -a
```

To switch to a remote experiment branch for the first time:

```bash
git switch --track origin/agent/worldsplat-vkitti2-rayfix
```

After the local branch exists, moving between branches is simply:

```bash
git switch agent/worldsplat-vkitti2-rayfix
git switch agent/worldsplat-vkitti2
git switch agent/worldsplat-studio
git switch agent/anchored-world-v0
git switch main
```

To update whichever branch you are currently on:

```bash
git pull
```

A safe general pattern before switching experiments is:

```bash
git status
git fetch origin
git switch <branch-name>
git pull
```

If `git status` shows local edits, commit or stash them before switching so Git does not mix your changes with another experiment.

---

# If you already cloned this repo earlier

From inside the existing clone:

```bash
git fetch origin
```

See the available remote branches:

```bash
git branch -r
```

Then create a local tracking branch, for example:

```bash
git switch --track origin/agent/worldsplat-vkitti2-rayfix
```

If Git says the branch already exists locally, just use:

```bash
git switch agent/worldsplat-vkitti2-rayfix
git pull
```

---

# Clone only one branch

If you only want one experiment and do not want the other branches locally:

```bash
git clone --branch agent/worldsplat-vkitti2-rayfix --single-branch \
  https://github.com/anttiluode/WorldModel.git
```

For the previous VKITTI2 control instead:

```bash
git clone --branch agent/worldsplat-vkitti2 --single-branch \
  https://github.com/anttiluode/WorldModel.git
```

For the original WorldSplat Studio:

```bash
git clone --branch agent/worldsplat-studio --single-branch \
  https://github.com/anttiluode/WorldModel.git
```

For the anchored-world work:

```bash
git clone --branch agent/anchored-world-v0 --single-branch \
  https://github.com/anttiluode/WorldModel.git
```

---

# Windows / CMD examples

The same commands work in Windows Command Prompt or PowerShell. For example:

```bat
cd E:\DocsHouse\932\WorldModel
git fetch origin
git switch agent/worldsplat-vkitti2-rayfix
git pull
python world_studio.py
```

If the branch has never been checked out locally:

```bat
git fetch origin
git switch --track origin/agent/worldsplat-vkitti2-rayfix
```

---

# Why the branches are kept separate

This repo is being used as an experimental ledger. A failed branch is still useful if it establishes a control.

For example:

```text
worldsplat-studio
    RGB-only / CelebA
    -> coherent rotatable face manifold
    -> wrong / unconstrained geometry

worldsplat-vkitti2
    RGB + simulator metric depth
    -> first outdoor scene attempt
    -> exposed a coordinate conflict and peripheral splat fog

worldsplat-vkitti2-rayfix
    same outdoor experiment
    -> ray position decoupled from depth
    -> strict A/B against the previous run
```

So branches are not merely software versions. They are often **experimental conditions**.

For the deeper technical note comparing this project with ordinary scene-specific 3D Gaussian Splatting, see `docs/WORLDSPLAT_VS_3DGS.md` on the `agent/worldsplat-vkitti2-rayfix` branch.

> **Do not hype. Do not lie. Just show.**
