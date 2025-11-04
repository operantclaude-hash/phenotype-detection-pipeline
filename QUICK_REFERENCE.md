# GitHub + Claude Code Quick Reference

**One-page guide for setting up and using your pipeline**

---

## 🚀 Initial Setup (One Time)

### 1. Create GitHub Repo
```bash
# On GitHub.com → New Repository
# Name: phenotype-detection-pipeline
# ✅ Public/Private
# ✅ Add README
# ✅ Add .gitignore (Python)
# ✅ Add LICENSE (MIT)
```

### 2. Clone and Setup
```bash
# Clone
git clone https://github.com/yourusername/phenotype-detection-pipeline.git
cd phenotype-detection-pipeline

# Copy your pipeline files here

# Run automated setup
chmod +x setup_repo.sh
./setup_repo.sh

# Review and commit
git commit -m "Initial setup: Phenotype detection pipeline"
git push
```

**Done! Ready for Claude Code.**

---

## 💻 Using Claude Code

### Start Session
```bash
cd phenotype-detection-pipeline
claude-code
```

### Common Tasks

| Task | Command |
|------|---------|
| **Customize for new experiment** | "Modify prepare_dataset.py to handle Drug_X with doses 0uM, 1uM, 10uM" |
| **Debug error** | "Getting KeyError: 'condition'. Can you fix?" |
| **Add feature** | "Add function to plot neuron trajectories" |
| **Update docs** | "Update README with my latest results" |
| **Write tests** | "Create unit tests for prepare_dataset.py" |
| **Refactor code** | "Make train_models.py more modular" |

### What Claude Code Can Do
- ✅ Read entire codebase
- ✅ Make targeted edits
- ✅ Create new files
- ✅ Run simple tests
- ✅ Write documentation
- ✅ Debug errors
- ✅ Suggest improvements

### What Claude Code Cannot Do
- ❌ Train models (use HPC)
- ❌ Process datasets (use compute cluster)
- ❌ Git push (you do manually)

---

## 📁 Directory Structure

```
phenotype-detection-pipeline/
├── src/                    # Your code
│   ├── prepare_dataset.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── analyze_results.py
├── docs/                   # Documentation
│   ├── QUICKSTART.md
│   ├── METHODOLOGY.md
│   └── USAGE.md
├── examples/               # Usage examples
├── config/                 # Config files
├── tests/                  # Unit tests
├── README.md              # Main docs
├── requirements.txt       # Dependencies
└── .gitignore            # Exclude data
```

---

## 🔄 Daily Workflow

### Making Changes
```bash
# 1. Create branch
git checkout -b feature/new-analysis

# 2. Use Claude Code
claude-code
"Help me add feature X"

# 3. Review changes
git diff

# 4. Commit
git add .
git commit -m "Add feature X"

# 5. Push
git push -u origin feature/new-analysis

# 6. Merge when ready
git checkout main
git merge feature/new-analysis
git push
```

---

## 🎯 Best Practices

### Do's ✅
- Commit often
- Use branches
- Write clear commit messages
- Keep data out of repo
- Document changes
- Test before pushing

### Don'ts ❌
- Don't commit data files (*.h5)
- Don't commit results (*.png, *.ckpt)
- Don't push to main directly
- Don't include passwords
- Don't commit huge files (>100MB)

---

## 🐛 Quick Fixes

### "Permission denied"
```bash
chmod +x script.sh
```

### "Not a git repository"
```bash
cd /path/to/phenotype-detection-pipeline
```

### "Merge conflict"
```bash
git status  # See conflicting files
# Edit files to resolve
git add .
git commit
```

### "CUDA out of memory"
```python
# In your script
--batch_size 16  # Reduce batch size
```

---

## 📚 Resources

- **GitHub Docs**: https://docs.github.com
- **Claude Code**: https://docs.claude.com/claude-code
- **Git Cheat Sheet**: https://training.github.com/downloads/github-git-cheat-sheet/

---

## 🎓 Example Session

```bash
$ cd phenotype-detection-pipeline
$ claude-code

You: "I need to add a new cell line called 'Mutant3' to the pipeline"

Claude Code: [Reads code]
"I'll update prepare_dataset.py to handle Mutant3. 
Should it be grouped with Control or APPV717I?"

You: "It's a new group on its own"

Claude Code: [Makes changes]
"Updated! Now you have 3 cell line groups. Want me to update 
the class structure to 9 classes (3 cell lines × T0/T16)?"

You: "Yes please"

Claude Code: [Updates train and evaluate scripts]
"Done! Updated:
- prepare_dataset.py: New cell line handling
- train_models.py: 9-class support
- evaluate_models.py: 9-class evaluation
Want me to commit these changes?"

You: "Yes"

Claude Code: [Creates commit]
"Committed: 'Add Mutant3 cell line support (9-class model)'
Ready to push when you are!"
```

---

## ✅ Checklist

Before first use:
- [ ] GitHub repo created
- [ ] Repo cloned locally
- [ ] Files organized with setup_repo.sh
- [ ] Initial commit pushed
- [ ] Claude Code installed
- [ ] Tested Claude Code can read repo

For each experiment:
- [ ] Create feature branch
- [ ] Customize scripts
- [ ] Test locally
- [ ] Commit changes
- [ ] Push to GitHub
- [ ] Merge to main when validated

---

## 🎉 You're Ready!

**Workflow**: 
Local Files → GitHub → Claude Code → Development → Validation → Production

**Key Insight**: 
GitHub = Version control + Collaboration
Claude Code = AI pair programmer

**Result**:
Faster development, better code, reproducible science! 🔬✨

---

*Print this page for quick reference while coding!*
