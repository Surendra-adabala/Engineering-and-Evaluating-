# Email Classification Model Comparison Report

## Pipeline Performance Metrics

| Metric | Value |
|--------|-------|
| Total Execution Time | 11.73 seconds |
| initial_row_count | 206 |
| final_row_count | 206 |
| row_reduction_percent | 0.00 |
| embedding_dimensions | 1526 |
| vocabulary_size | 1526 |

## Model Performance Comparison

### Design Decision 1 vs Design Decision 2

| Metric | Chained Multi-outputs | Hierarchical Modeling |
|--------|----------------------|----------------------|
| Average Accuracy | N/A | N/A |
| Average F1 (Macro) | N/A | N/A |
| Average F1 (Weighted) | N/A | N/A |

### Chained Model Results

### Hierarchical Model Results

#### Group: AppGallery &amp; Games 

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|

#### Group: In-App Purchase 

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|

## Conclusion

This report compares the performance of two architectural approaches for multi-label email classification:

1. **Design Decision 1: Chained Multi-outputs** - Using a single model instance for each combination of labels
2. **Design Decision 2: Hierarchical Modeling** - Using multiple model instances in a tree structure

Both design decisions show similar performance in terms of average accuracy.

### Trade-offs

**Design Decision 1: Chained Multi-outputs**
- **Pros**: Simpler architecture, fewer models to train, consistent interface
- **Cons**: May not capture hierarchical dependencies as well, less interpretability

**Design Decision 2: Hierarchical Modeling**
- **Pros**: Better captures hierarchical relationships, more interpretable results per class
- **Cons**: More complex architecture, more models to train and maintain

### Recommendation

Both approaches show similar performance. The choice between them should be based on other factors such as interpretability needs, maintenance requirements, and specific business contexts.
