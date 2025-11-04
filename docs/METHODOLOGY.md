# Methodology: Artifact-Free Phenotype Detection

**Scientific rationale for the 6-class classification approach**

---

## 🔬 The Problem: Technical Artifacts in Image-Based Screening

### Background

High-content imaging screens generate thousands of images across multi-well plates over time. Deep learning models can achieve high accuracy classifying these images, but often learn **technical artifacts** rather than biological signal:

1. **Positional effects**: Cells in corner wells may look different from center wells
2. **Illumination gradients**: Uneven lighting across the plate
3. **Batch effects**: Different imaging sessions have systematic differences  
4. **Temporal artifacts**: Autofluorescence changes, photobleaching
5. **Experimental design leakage**: Model learns which wells will receive treatment

### The Critical Flaw in Standard Approaches

Consider a typical temporal experiment:
- **T0**: Baseline imaging (pre-treatment)
- **Treatment**: Apply drug/stimulation
- **T16**: Endpoint imaging (post-treatment)

**Standard 8-class model:**
```
Classes:
1. WT_0ms_T0
2. WT_0ms_T16
3. WT_10000ms_T0     ← These should be identical!
4. WT_10000ms_T16
5. APP_0ms_T0
6. APP_0ms_T16
7. APP_10000ms_T0    ← These should be identical!
8. APP_10000ms_T16
```

**The artifact**: At T0, cells haven't been treated yet! The "0ms" and "10000ms" groups are biologically identical. Any model accuracy distinguishing them must come from:
- Well positions (treatment assigned by location)
- Pre-existing imaging differences
- Experimental design knowledge

**Evidence from our data:**
- Halo channel: 59% accuracy on `WT_10000ms_T0` vs 37% on `WT_0ms_T0`
- This 22% gap is **impossible biologically** at T0
- Model learned which wells would later be stimulated!

---

## ✅ The Solution: Collapse Pre-Treatment Timepoints

### 6-Class Design

**Rationale**: Prevent the model from learning pre-treatment well assignments.

**Implementation**:
```python
if timepoint == 'T0':
    # Cells haven't been treated - merge stimulation groups
    label = f"{cell_line}_T0"
else:  # T16
    # Phenotype has emerged - keep groups separate
    label = f"{cell_line}_{stimulation}_T16"
```

**Resulting classes:**
```
1. WT_T0              ← Merged from WT_0ms_T0 + WT_10000ms_T0
2. APP_T0             ← Merged from APP_0ms_T0 + APP_10000ms_T0
3. WT_0ms_T16         ← Distinct (no treatment effect)
4. WT_10000ms_T16     ← Distinct (treatment effect)
5. APP_0ms_T16        ← Distinct (no treatment effect)
6. APP_10000ms_T16    ← Distinct (treatment effect)
```

### Why This Works

**Forces model to learn biology**:
- At T0: Can only distinguish by **cell line** (WT vs APP)
- At T16: Can distinguish by **cell line AND treatment**
- Cannot memorize well positions or future treatment assignments

**Expected accuracy patterns**:
```
Good model:
  WT_T0: 50-60% (distinguishes from APP_T0)
  APP_T0: 50-60% (distinguishes from WT_T0)
  WT_10000ms_T16: 50-70% (treatment phenotype visible)
  WT_0ms_T16: 30-50% (no treatment, harder to classify)
```

**Red flag patterns**:
```
Bad model (still learning artifacts):
  WT_T0: 80% but confused internally
  Confusion matrix shows WT_T0 split by future treatment
```

---

## 📊 Statistical Framework

### Performance Metrics

**1. Overall Accuracy**
- Random baseline: 1/6 = 16.7%
- Acceptable: >35% (2× baseline)
- Good: >45% (2.7× baseline)
- Excellent: >55% (3.3× baseline)

**2. Per-Class Accuracy**
Diagonal elements of confusion matrix:
```python
class_acc = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
```

**3. Temporal Delta**
Change in accuracy from T0 to T16:
```python
t0_acc = mean(class_acc[classes with T0])
t16_acc = mean(class_acc[classes with T16])
temporal_delta = t16_acc - t0_acc
```
- Positive delta: Phenotype emerges over time ✓
- Zero/negative delta: No temporal signal ✗

**4. Treatment Separability**
Within each cell line at T16:
```python
# For WT:
wt_treatment_gap = abs(acc[WT_10000ms_T16] - acc[WT_0ms_T16])
```
- Large gap (>10%): Treatment effect strong ✓
- Small gap (<5%): Weak treatment effect ✗

**5. Artifact Contribution**
Compare 8-class vs 6-class:
```python
artifact_contribution = acc_8class - acc_6class
```
- Positive: 8-class was learning artifacts ✗
- Zero/negative: 6-class as good or better ✓

---

## 🧬 Biological Interpretation

### Cell Line Effects (T0 Separation)

**High T0 accuracy for cell line separation** indicates:
- ✅ Baseline morphological differences exist
- ✅ Genetic perturbation has visible phenotype even without treatment
- ✅ Model can distinguish cell types

**Low T0 accuracy** indicates:
- Cell lines are morphologically similar at baseline
- Genetic perturbation has subtle/no basal phenotype
- Treatment may be required to unmask differences

### Treatment Effects (T16 Separation)

**High T16 treatment separability** indicates:
- ✅ Treatment induces robust phenotypic changes
- ✅ Morphological or reporter signal is strong
- ✅ Effect is detectable by imaging

**Low T16 separability** indicates:
- Treatment has weak or no effect on this readout
- Wrong timepoint (may need earlier/later)
- Wrong channel (try different markers)
- Subtle changes requiring more sensitive methods

### Channel Comparison

**RFP1 (Morphology) typically better for:**
- Structural changes (neurite length, branching)
- Cell shape alterations
- Cytoskeletal reorganization
- Later-stage phenotypes

**Halo (Reporter) typically better for:**
- Protein localization changes
- Early molecular events
- Subtle biochemical changes
- Rapid responses

---

## 🔍 Confusion Matrix Analysis

### Reading the Matrix

**Normalized confusion matrix** shows:
- **Rows**: True labels (ground truth)
- **Columns**: Predicted labels (model output)
- **Diagonal**: Correct predictions (per-class accuracy)
- **Off-diagonal**: Misclassifications

### Interpretation Patterns

**Pattern 1: Strong diagonal at T16**
```
                  Pred_WT_0ms  Pred_WT_10000ms
True_WT_0ms_T16       0.65          0.15
True_WT_10000ms_T16   0.10          0.70
```
✅ **Good**: Treatment groups are separable

**Pattern 2: Weak diagonal at T16**
```
                  Pred_WT_0ms  Pred_WT_10000ms
True_WT_0ms_T16       0.35          0.40
True_WT_10000ms_T16   0.45          0.35
```
⚠️ **Concerning**: Treatment has weak/no effect

**Pattern 3: T0 confusion stays within cell line**
```
                  Pred_WT_T0  Pred_APP_T0
True_WT_T0            0.60       0.25
True_APP_T0           0.20       0.65
```
✅ **Good**: Cell lines separate, no artifact leakage

**Pattern 4: Cross-contamination at T0**
```
                  Pred_WT_0ms  Pred_WT_10000ms
True_WT_T0            0.40          0.30
```
⚠️ **Red flag**: Model split T0 by future treatment (artifact!)

---

## 📈 Experimental Design Considerations

### Sample Size

**Minimum recommended:**
- ≥1000 cells per class
- ≥5000 cells total
- ≥3 biological replicates

**Better performance with:**
- ≥2000 cells per class
- ≥10000 cells total
- ≥5 biological replicates

### Class Balance

**Imbalanced classes handled by:**
1. Stratified train/val/test splits
2. Class weights in loss function
3. Oversampling minority classes

**Check balance:**
```python
df.groupby('label_6class').size()
```

**Severe imbalance** (>5:1 ratio):
- Consider resampling
- Use weighted loss
- Report per-class metrics

### Data Quality

**Critical quality checks:**
1. **Focus**: Out-of-focus images reduce accuracy
2. **Exposure**: Over/under-exposed images cause artifacts
3. **Drift**: Stage drift creates positional artifacts
4. **Debris**: Dead cells/debris confound signal
5. **Tracking**: Incorrect neuron tracking mislabels data

**Preprocessing helps:**
- Normalize intensity across wells
- Remove outliers (too bright/dark)
- Filter by quality metrics (e.g., texture)

---

## 🎯 Model Selection

### Architecture Choices

**ResNet18** (default):
- ✅ Fast training (~10 min)
- ✅ Good baseline
- ✅ Sufficient for most tasks
- ⚠️ May underfit complex phenotypes

**ResNet50**:
- ✅ More capacity
- ✅ Better for subtle phenotypes
- ⚠️ Slower training (~20 min)
- ⚠️ May overfit small datasets

**EfficientNet**:
- ✅ State-of-art efficiency
- ✅ Great accuracy/speed tradeoff
- ⚠️ Requires more tuning

### Hyperparameters

**Learning rate**:
- Default: 0.001
- Too high: Training unstable
- Too low: Slow convergence

**Batch size**:
- Default: 32
- Larger (64): Faster, less noisy, needs more memory
- Smaller (16): Slower, more noisy, fits on small GPUs

**Epochs**:
- Max: 100
- Early stopping: 15 epochs patience
- Typically stops: 40-60 epochs

---

## 🔬 Validation Strategy

### Data Splits

**Stratified by neuron ID**:
- Ensures neurons don't appear in multiple splits
- Prevents data leakage
- Models generalize to new cells

**Split ratios**:
- Train: 70% (neurons)
- Validation: 15% (neurons)
- Test: 15% (neurons)

**Why by neuron, not image?**
- Two timepoints per neuron (T0, T16)
- Same neuron in train and test = data leakage
- Model memorizes specific cells

### Cross-Validation

For small datasets (<5000 cells):
```python
# 5-fold cross-validation
for fold in range(5):
    train_model(fold)
    evaluate(fold)
report_mean_and_std()
```

---

## 📚 References & Further Reading

### Key Concepts

1. **Data leakage in ML**: Model has access to information not available at prediction time
2. **Stratified sampling**: Maintaining class proportions across splits
3. **Confusion matrix**: Visualization of classification performance
4. **Transfer learning**: Using pretrained ImageNet weights

### Recommended Papers

1. Caicedo et al. (2017) "Data-analysis strategies for image-based cell profiling"
2. Chandrasekaran et al. (2021) "Image-based profiling for drug discovery"
3. Seal et al. (2022) "DeepProfiler: Deep learning for image-based profiling"

### Common Pitfalls

1. **Not accounting for well effects**: Always check positional artifacts
2. **Ignoring temporal structure**: Neuron tracking creates dependencies
3. **Overfitting to technical variation**: Models memorize noise
4. **Selection bias**: Analyzing only "good looking" cells
5. **P-hacking**: Testing many conditions without correction

---

## ✅ Validation Checklist

Before publishing results:

- [ ] Checked confusion matrix for artifact patterns
- [ ] Compared 8-class vs 6-class (if applicable)
- [ ] Verified T0 doesn't separate by future treatment
- [ ] Confirmed T16 separates by treatment
- [ ] Validated on held-out test set
- [ ] Checked per-class accuracies
- [ ] Compared multiple channels
- [ ] Assessed biological plausibility
- [ ] Tested on independent experiment (if available)
- [ ] Reported random baseline comparison

---

## 🎓 Summary

**The 6-class approach ensures:**
1. Models learn biological signal, not experimental design
2. Pre-treatment timepoints can't leak treatment information
3. Phenotypes must emerge after treatment to be detected
4. Results are interpretable and scientifically valid

**Key insight**: Lower accuracy with honest classification is better than high accuracy from cheating. The goal is to **detect real biology**, not to optimize metrics.

**Success = Detecting biological phenotypes with confidence that the signal is real.**

---

*For questions on methodology, see QUICKSTART.md for implementation or README.md for overview.*
