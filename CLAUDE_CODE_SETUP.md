# Setting Up for Claude Code + GitHub

**Step-by-step guide to organize your pipeline for GitHub and use with Claude Code**

---

## 🎯 Workflow Overview

```
Your Code → GitHub → Claude Code → Development
   ↓           ↓          ↓            ↓
  Push     Version    Pull/Edit    Test/Run
          Control
```

---

## 📝 Step 1: Create GitHub Repository

### On GitHub.com:

1. **Create new repository**
   - Go to https://github.com/new
   - Name: `phenotype-detection-pipeline` (or your choice)
   - Description: "Artifact-free deep learning for biological phenotype detection"
   - ✅ Public or Private (your choice)
   - ✅ Add README (we'll replace it)
   - ✅ Add .gitignore → Python
   - ✅ Add license → MIT

2. **Clone to your local machine**
   ```bash
   git clone https://github.com/yourusername/phenotype-detection-pipeline.git
   cd phenotype-detection-pipeline
   ```

---

## 📁 Step 2: Organize Your Files

### Recommended Structure:

```bash
# Create directory structure
mkdir -p src docs examples config tests

# Move your pipeline files
mv 1_prepare_dataset.py src/prepare_dataset.py
mv 2_train_models.py src/train_models.py
mv 3_evaluate_models.py src/evaluate_models.py
mv 4_analyze_results.py src/analyze_results.py

# Move documentation
mv README.md docs/OVERVIEW.md
mv QUICKSTART.md docs/QUICKSTART.md
mv METHODOLOGY.md docs/METHODOLOGY.md
mv PACKAGE_MANIFEST.md docs/USAGE.md

# Copy example files
mv example_commands.sh examples/
mv expected_outputs.md examples/
```

### Create __init__.py:

```bash
cat > src/__init__.py << 'EOF'
"""
Artifact-Free Phenotype Detection Pipeline
"""
__version__ = "1.0.0"
EOF
```

---

## 📦 Step 3: Add Essential Files

### Add requirements.txt:
```bash
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
h5py>=3.8.0
Pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
tensorboard>=2.13.0
EOF
```

### Add .gitignore:
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pth
*.ckpt

# Data - Don't commit large files!
data/
dataset/
*.h5
*.hdf5

# Results
results/
outputs/
evaluation/
analysis/
*.png
*.npy

# Environment
.env
venv/
EOF
```

### Update main README:
```bash
# Use the GitHub-ready README
cp README_GITHUB.md README.md
```

---

## 🚀 Step 4: Initial Commit

```bash
# Stage all files
git add .

# Commit
git commit -m "Initial commit: Artifact-free phenotype detection pipeline

- Add 4 main scripts (prepare, train, evaluate, analyze)
- Add comprehensive documentation
- Add examples and usage guides
- Add requirements.txt and .gitignore
"

# Push to GitHub
git push origin main
```

---

## 💻 Step 5: Use with Claude Code

### Option A: Clone and Work Locally

```bash
# In your terminal
cd ~/projects
git clone https://github.com/yourusername/phenotype-detection-pipeline.git
cd phenotype-detection-pipeline

# Open with Claude Code
claude-code
```

Then in Claude Code:
```
"Open the phenotype-detection-pipeline repository and help me 
customize prepare_dataset.py for my new experiment"
```

### Option B: Direct GitHub Integration

In Claude Code terminal:
```bash
# Tell Claude Code about your repo
"I have a GitHub repo at https://github.com/yourusername/phenotype-detection-pipeline. 
Can you help me add a new feature to extract different metadata fields?"
```

Claude Code can:
- ✅ Read your GitHub repo
- ✅ Make changes to files
- ✅ Test changes locally
- ✅ Help you commit/push changes

---

## 🔄 Step 6: Development Workflow

### Typical Claude Code Session:

1. **Start Claude Code**
   ```bash
   cd phenotype-detection-pipeline
   claude-code
   ```

2. **Ask Claude to help**
   ```
   "I need to modify prepare_dataset.py to handle a new condition 
   called 'Drug_X' with doses '0uM', '1uM', '10uM'. Can you update 
   the code?"
   ```

3. **Claude Code will:**
   - Read the current code
   - Understand the structure
   - Make the changes
   - Test if possible
   - Show you the diff

4. **Review and commit**
   ```bash
   git diff  # Review changes
   git add src/prepare_dataset.py
   git commit -m "Add Drug_X condition support"
   git push
   ```

---

## 🌿 Step 7: Branching Strategy

### For Each Experiment:

```bash
# Create experiment branch
git checkout -b experiment/plate3_analysis

# Make changes with Claude Code
# "Help me modify the training script for plate 3 data"

# Commit and push
git add .
git commit -m "Customize for plate 3 analysis"
git push -u origin experiment/plate3_analysis

# After validation, merge to main
git checkout main
git merge experiment/plate3_analysis
git push
```

---

## 📊 Step 8: Share Results

### Add Results to Docs (optional):

```bash
# Create results branch
git checkout -b results/experiment1

# Add figures (small PNGs only!)
mkdir -p docs/results
cp evaluation/RFP1/confusion_matrix_normalized.png docs/results/
cp analysis/6class_detailed_analysis.png docs/results/

# Update README with results
# Use Claude Code: "Add my results to the README"

git add docs/results/
git commit -m "Add experiment 1 results"
git push
```

---

## 🤖 Claude Code Use Cases

### Perfect for:

1. **Customization**
   ```
   "Modify prepare_dataset.py to extract drug concentration 
   from well IDs like 'A1_100nM'"
   ```

2. **Debugging**
   ```
   "I'm getting KeyError: 'label_6class'. Can you help debug?"
   ```

3. **New Features**
   ```
   "Add a function to visualize individual neuron trajectories 
   over time"
   ```

4. **Testing**
   ```
   "Create unit tests for the prepare_dataset.py functions"
   ```

5. **Documentation**
   ```
   "Update the README with information about my new features"
   ```

### Claude Code Can:
- ✅ Read entire codebase
- ✅ Understand context
- ✅ Make targeted changes
- ✅ Run tests
- ✅ Commit changes
- ✅ Explain code

### Claude Code Cannot:
- ❌ Train models (too computationally intensive)
- ❌ Process large datasets (use HPC for that)
- ❌ Replace your domain expertise

---

## 🔐 Best Practices

### Do's:
- ✅ Commit often with clear messages
- ✅ Use branches for experiments
- ✅ Keep data out of repo (use .gitignore)
- ✅ Document changes in commit messages
- ✅ Use issues to track TODOs
- ✅ Add example config files

### Don'ts:
- ❌ Commit data files (*.h5, *.hdf5)
- ❌ Commit model weights (*.ckpt)
- ❌ Commit results (*.png, *.npy)
- ❌ Push directly to main (use branches)
- ❌ Include API keys or passwords
- ❌ Commit huge files (>100 MB)

---

## 📂 Final Directory Structure

```
phenotype-detection-pipeline/
├── README.md                    # Main documentation (GitHub landing page)
├── LICENSE                      # MIT license
├── requirements.txt             # Python dependencies
├── .gitignore                   # Exclude data/results
├── setup.py                     # Optional: Make installable
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── prepare_dataset.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── analyze_results.py
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md
│   ├── METHODOLOGY.md
│   ├── USAGE.md
│   └── images/                  # Screenshots for README
│
├── examples/                    # Example usage
│   ├── example_workflow.sh
│   └── example_config.yaml
│
├── config/                      # Configuration files
│   └── default_config.yaml
│
└── tests/                       # Unit tests (optional)
    └── test_prepare_dataset.py
```

---

## 🎯 Example Claude Code Conversation

```
You: "I just cloned my phenotype-detection-pipeline repo. 
     Can you help me adapt it for a new experiment?"

Claude Code: "I can see your pipeline! What's different about 
              the new experiment?"

You: "It uses different cell lines: 'WT', 'KO1', 'KO2' instead 
     of Control and APPV717I"

Claude Code: "I'll update prepare_dataset.py to handle the new 
              cell lines. I'll also need to update the class 
              names in train_models.py and evaluate_models.py. 
              Let me make those changes..."
              
              [Makes changes]
              
              "Done! I've updated:
              1. extract_condition_info() to parse new cell lines
              2. CLASS_CONFIG class names
              3. Documentation strings
              
              Want me to commit these changes?"

You: "Yes, and can you also update the README?"

Claude Code: "Sure! Committing and updating docs..."
```

---

## 🚀 Quick Setup Script

Save this as `setup_repo.sh`:

```bash
#!/bin/bash
# Quick setup for phenotype detection pipeline

# Create structure
mkdir -p src docs examples config tests

# Move files (adjust paths as needed)
mv *_prepare_dataset.py src/prepare_dataset.py
mv *_train_models.py src/train_models.py
mv *_evaluate_models.py src/evaluate_models.py
mv *_analyze_results.py src/analyze_results.py

# Create __init__.py
echo '"""Phenotype Detection Pipeline"""' > src/__init__.py
echo '__version__ = "1.0.0"' >> src/__init__.py

# Initial commit
git add .
git commit -m "Initial commit: Pipeline setup"
git push

echo "✅ Setup complete! Ready for Claude Code."
```

---

## 📚 Resources

- **GitHub Docs**: https://docs.github.com
- **Claude Code**: https://docs.claude.com/claude-code
- **Git Tutorial**: https://git-scm.com/book

---

## ✅ Checklist

Before using with Claude Code:

- [ ] Created GitHub repository
- [ ] Organized files in proper structure
- [ ] Added requirements.txt
- [ ] Added .gitignore
- [ ] Made initial commit
- [ ] Pushed to GitHub
- [ ] Cloned locally
- [ ] Tested Claude Code can read repo
- [ ] Ready to iterate!

---

**🎉 You're now set up for efficient development with Claude Code + GitHub!**

Your code is version controlled, shareable, and ready for collaborative development with AI assistance.
