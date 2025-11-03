# 🚀 Quick Environment Setup Reference

## Three Methods - Choose One:

### ✅ Method 1: RECOMMENDED (Full Conda)
```bash
conda env create -f environment_ovnpu.yml
conda activate ovnpu
```
**Use when:** Setting up on any new system (Windows/Linux/Mac)

---

### ⚡ Method 2: MOST PRECISE (Explicit Conda)
```bash
conda create --name ovnpu --file conda_explicit_ovnpu.txt
conda activate ovnpu
```
**Use when:** Need exact binary replication on same platform

---

### 🐍 Method 3: PIP ONLY
```bash
conda create -n ovnpu python=3.11 -y
conda activate ovnpu
pip install -r requirements_ovnpu_frozen.txt
```
**Use when:** No conda available or Docker environment

---

## 🔄 Update Environment Files
```bash
conda activate ovnpu
conda env export > environment_ovnpu.yml
pip freeze > requirements_ovnpu_frozen.txt
conda list --explicit > conda_explicit_ovnpu.txt
```

---

## ✅ Test Installation
```bash
conda activate ovnpu
python -c "import torch, transformers, openvino; print('✓ All packages loaded!')"
```

---

## 📁 Files in This Folder

| File | Method | Best For |
|------|--------|----------|
| `environment_ovnpu.yml` | Method 1 | Cross-platform, easiest |
| `conda_explicit_ovnpu.txt` | Method 2 | Exact replication, same platform |
| `requirements_ovnpu_frozen.txt` | Method 3 | Pip-only, containers |
| `ENVIRONMENT_SETUP.md` | All | Detailed guide |
| `QUICK_SETUP.md` | All | This reference card |

---

**See ENVIRONMENT_SETUP.md for detailed instructions and troubleshooting**
