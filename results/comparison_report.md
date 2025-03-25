# Email Classification Model Comparison Report

## Pipeline Performance Metrics

| Metric | Value |
|--------|-------|
| Total Execution Time | 4.46 seconds |
| initial_row_count | 206 |
| final_row_count | 206 |
| row_reduction_percent | 0.00 |
| embedding_dimensions | 1526 |
| vocabulary_size | 1526 |

## Model Performance Comparison

### Design Decision 1 vs Design Decision 2

| Metric | Chained Multi-outputs | Hierarchical Modeling |
|--------|----------------------|----------------------|
| Average Accuracy | 77.18% | N/A |
| Average F1 (Macro) | 53.50% | N/A |
| Average F1 (Weighted) | 75.30% | N/A |

### Chained Model Results

#### Group: AppGallery &amp; Games 

| Level | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| Type 2 | 72.00% | 61.83% | 70.91% |

#### Group: In-App Purchase 

| Level | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| Type 2 | 82.35% | 45.16% | 79.70% |

### Hierarchical Model Results

## Conclusion

This report compares the performance of two architectural approaches for multi-label email classification:

1. **Design Decision 1: Chained Multi-outputs** - Using a single model instance for each combination of labels
2. **Design Decision 2: Hierarchical Modeling** - Using multiple model instances in a tree structure

Based on the average accuracy, **Design Decision 1: Chained Multi-outputs** performs better. It achieves an average accuracy of 77.18% compared to 0.00% for the hierarchical approach.

### Trade-offs

**Design Decision 1: Chained Multi-outputs**
- **Pros**: Simpler architecture, fewer models to train, consistent interface
- **Cons**: May not capture hierarchical dependencies as well, less interpretability

**Design Decision 2: Hierarchical Modeling**
- **Pros**: Better captures hierarchical relationships, more interpretable results per class
- **Cons**: More complex architecture, more models to train and maintain

### Recommendation

For this specific email classification task, the **Chained Multi-outputs** approach is recommended due to its better performance and simpler architecture.
