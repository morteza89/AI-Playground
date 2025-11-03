# Environment Setup Guide for AI-Benchmarking

This guide provides multiple methods to replicate the exact `ovnpu` environment on other systems.

## 📦 Available Environment Files

Three files are provided for different installation scenarios:

1. **`environment_ovnpu.yml`** - Full conda environment (RECOMMENDED)
2. **`requirements_ovnpu_frozen.txt`** - Pip-only with exact versions
3. **`conda_explicit_ovnpu.txt`** - Explicit conda URLs (most precise)

---

## 🎯 Method 1: Full Conda Environment (RECOMMENDED)

**Best for:** Complete replication including all conda and pip packages

### Installation Steps:

```bash
# Create environment from YAML file
conda env create -f environment_ovnpu.yml

# Activate the environment
conda activate ovnpu

# Verify installation
python --version
pip list
```

### Advantages:
- ✅ Includes both conda and pip packages
- ✅ Preserves environment name
- ✅ Cross-platform compatible
- ✅ Easiest to use

### Update Existing Environment:
```bash
conda env update -f environment_ovnpu.yml --prune
```

---

## 🔧 Method 2: Explicit Conda Specification (MOST PRECISE)

**Best for:** Exact binary replication on same OS/architecture

### Installation Steps:

```bash
# Create environment from explicit spec
conda create --name ovnpu --file conda_explicit_ovnpu.txt

# Activate the environment
conda activate ovnpu

# Verify installation
conda list
```

### Advantages:
- ✅ Exact binary packages (bit-for-bit identical)
- ✅ No dependency resolution needed (faster)
- ✅ Guaranteed to work on same platform

### Limitations:
- ⚠️ Platform-specific (Windows/Linux/Mac)
- ⚠️ Architecture-specific (x86_64/ARM)

---

## 🐍 Method 3: Pip Requirements Only

**Best for:** pip-based workflows or Docker containers

### Installation Steps:

```bash
# Create a new conda environment (Python 3.11 recommended)
conda create -n ovnpu python=3.11 -y
conda activate ovnpu

# Install all packages with exact versions
pip install -r requirements_ovnpu_frozen.txt

# Verify installation
pip list
```

### Advantages:
- ✅ Works in any Python environment
- ✅ Good for Docker/containers
- ✅ No conda required

### Limitations:
- ⚠️ Missing conda-specific packages
- ⚠️ May have system dependency issues

---

## 🚀 Quick Start for New Systems

### On Windows:

```powershell
# Clone or copy the AI-Benchmarking folder
cd AI-Benchmarking

# Method 1: Full environment
conda env create -f environment_ovnpu.yml
conda activate ovnpu

# Test installation
python run_dynamic_5benchmark_dataset_test_general_OVmodel.py
```

### On Linux/Mac:

```bash
# Clone or copy the AI-Benchmarking folder
cd AI-Benchmarking

# Method 1: Full environment
conda env create -f environment_ovnpu.yml
conda activate ovnpu

# Test installation
python run_dynamic_5benchmark_dataset_test_general_OVmodel.py
```

---

## 🔄 Updating Environment Files

When you add new packages to your environment, update the files:

```bash
# Activate environment
conda activate ovnpu

# Update all three files
conda env export > environment_ovnpu.yml
pip freeze > requirements_ovnpu_frozen.txt
conda list --explicit > conda_explicit_ovnpu.txt
```

---

## 📋 Key Packages Included

The environment includes:

- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace models
- **OpenVINO** - Intel optimization toolkit
- **Optimum[openvino]** - OpenVINO integration
- **Datasets** - HuggingFace datasets library
- **NumPy, Pandas** - Data manipulation
- **And many more dependencies...**

---

## ⚠️ Troubleshooting

### Issue: Environment creation fails

**Solution:**
```bash
# Update conda first
conda update -n base conda

# Clear conda cache
conda clean --all

# Try again
conda env create -f environment_ovnpu.yml
```

### Issue: Package conflicts

**Solution:**
```bash
# Remove environment and recreate
conda env remove -n ovnpu
conda env create -f environment_ovnpu.yml
```

### Issue: Platform incompatibility (explicit spec)

**Solution:** Use Method 1 (YAML) instead - it resolves dependencies for your platform

---

## 🎓 Best Practices

1. **Always activate environment before running scripts:**
   ```bash
   conda activate ovnpu
   ```

2. **Keep environment files updated:**
   - Update after installing new packages
   - Version control these files (already in git)

3. **Test on new system:**
   ```bash
   python -c "import torch, transformers, openvino; print('Success!')"
   ```

4. **Use virtual environments:**
   - Never install directly in base environment
   - Keeps projects isolated

---

## 📚 Additional Resources

- [Conda Environment Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [OpenVINO Installation Guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html)
- [HuggingFace Installation](https://huggingface.co/docs/transformers/installation)

---

## 🆘 Support

If you encounter issues:

1. Check Python version: `python --version` (should be 3.11+)
2. Check conda version: `conda --version`
3. Verify GPU drivers if using iGPU/NPU
4. Review the log files for detailed error messages

---

**Environment Last Updated:** November 3, 2025
**Python Version:** 3.11+
**Platform:** Windows (cross-platform compatible with Method 1)
