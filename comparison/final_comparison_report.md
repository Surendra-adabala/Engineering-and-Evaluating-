====================================================================================================
DETAILED COMPARISON OF RESULTS - METRICS ANALYSIS
====================================================================================================



MODEL STRUCTURE COMPARISON
------------------------------------------------------------------------------------------
+--------------------+--------------------------------+--------------------------+
|       Metric       |     Chained Multi-outputs      |  Hierarchical Modeling   |
+--------------------+--------------------------------+--------------------------+
|    Total Models    |               28               |            28            |
|      Approach      |  Single model per group with   |   Multiple models in a   |
|                    |    multiple target outputs     |  hierarchical structure  |
|  Training Pattern  | Joint prediction of all labels | Separate models for each |
|                    |                                |    label combination     |
|  Model Complexity  |             Lower              |          Higher          |
| Maintenance Effort |             Lower              |          Higher          |
+--------------------+--------------------------------+--------------------------+



EXECUTION TIME COMPARISON
------------------------------------------------------------------------------------------
+----------------------+-----------------------+-----------------------+
|        Metric        | Chained Multi-outputs | Hierarchical Modeling |
+----------------------+-----------------------+-----------------------+
| Total Execution Time |      5.78 seconds     |     13.80 seconds     |
|    Relative Speed    |     1x (baseline)     |      2.39x slower     |
|    Models Trained    |           28          |           28          |
|    Time per Model    |     0.21 sec/model    |     0.49 sec/model    |
+----------------------+-----------------------+-----------------------+



MODEL ACCURACY METRICS
------------------------------------------------------------------------------------------
+-------------------------+-----------------------+----------+-----------+--------+----------+
|          Group          |          Type         | Accuracy | Precision | Recall | F1 Score |
+-------------------------+-----------------------+----------+-----------+--------+----------+
| AppGallery &amp; Games  |    Chained (Type 2)   |  72.00%  |   70.67%  | 72.00% |  70.91%  |
|     In-App Purchase     |    Chained (Type 2)   |  82.35%  |   77.21%  | 82.35% |  79.70%  |
|   AppGallery & Games    | Hierarchical (Type 2) |  72.00%  |   70.67%  | 72.00% |  70.91%  |
|     In-App Purchase     | Hierarchical (Type 2) |  82.35%  |   77.21%  | 82.35% |  79.70%  |
+-------------------------+-----------------------+----------+-----------+--------+----------+



DETAILED CLASSIFICATION REPORT
-----------------------------

CHAINED APPROACH CLASSIFICATION METRICS:

  Group: AppGallery &amp; Games 
    Type 2 Classification:
      Accuracy: 72.00%
      Precision: 70.67%
      Recall: 72.00%
      F1 Score (weighted): 70.91%

      Per-Class Metrics:
+---------------+-----------+--------+----------+---------+
|     Class     | Precision | Recall | F1 Score | Support |
+---------------+-----------+--------+----------+---------+
|     Others    |   83.33%  | 71.43% |  76.92%  |   7.0   |
| Problem/Fault |   75.00%  | 85.71% |  80.00%  |   14.0  |
|   Suggestion  |   33.33%  | 25.00% |  28.57%  |   4.0   |
+---------------+-----------+--------+----------+---------+
  Group: In-App Purchase 
    Type 2 Classification:
      Accuracy: 82.35%
      Precision: 77.21%
      Recall: 82.35%
      F1 Score (weighted): 79.70%

      Per-Class Metrics:
+---------------+-----------+--------+----------+---------+
|     Class     | Precision | Recall | F1 Score | Support |
+---------------+-----------+--------+----------+---------+
| Problem/Fault |   0.00%   | 0.00%  |  0.00%   |   2.0   |
|   Suggestion  |   87.50%  | 93.33% |  90.32%  |   15.0  |
+---------------+-----------+--------+----------+---------+

HIERARCHICAL APPROACH CLASSIFICATION METRICS:

  Group: AppGallery & Games 
    Type 2 Classification:
      Accuracy: 72.00%
      Precision: 70.67%
      Recall: 72.00%
      F1 Score (weighted): 70.91%

      Per-Class Metrics:
+---------------+-----------+--------+----------+---------+
|     Class     | Precision | Recall | F1 Score | Support |
+---------------+-----------+--------+----------+---------+
|     Others    |   83.33%  | 71.43% |  76.92%  |   7.0   |
| Problem/Fault |   75.00%  | 85.71% |  80.00%  |   14.0  |
|   Suggestion  |   33.33%  | 25.00% |  28.57%  |   4.0   |
+---------------+-----------+--------+----------+---------+
  Group: In-App Purchase 
    Type 2 Classification:
      Accuracy: 82.35%
      Precision: 77.21%
      Recall: 82.35%
      F1 Score (weighted): 79.70%

      Per-Class Metrics:
+---------------+-----------+--------+----------+---------+
|     Class     | Precision | Recall | F1 Score | Support |
+---------------+-----------+--------+----------+---------+
| Problem/Fault |   0.00%   | 0.00%  |  0.00%   |   2.0   |
|   Suggestion  |   87.50%  | 93.33% |  90.32%  |   15.0  |
+---------------+-----------+--------+----------+---------+



MOST IMPORTANT EMAIL FEATURES
------------------------------------------------------------------------------------------
+------------------------------+--------------------+-------------------------+
|           Feature            | Chained Importance | Hierarchical Importance |
+------------------------------+--------------------+-------------------------+
|   Word frequency: problem    |        High        |           High          |
| Word frequency: subscription |        High        |          Medium         |
|         Email length         |       Medium       |           Low           |
|   Word frequency: payment    |        High        |           High          |
|    Word frequency: error     |       Medium       |           High          |
+------------------------------+--------------------+-------------------------+



====================================================================================================
FINAL ANALYSIS AND RECOMMENDATION
====================================================================================================


+------+--------------------------------------+-----------------------------+
|      |        Chained Multi-outputs         |    Hierarchical Modeling    |
+------+--------------------------------------+-----------------------------+
| PROS |        • Simpler architecture        |  • Better interpretability  |
|      |       • Fewer models to train        |   • Class-specific models   |
|      |          • Faster execution          |   • Clearer error tracing   |
|      |         • Easier maintenance         |     • Flexible structure    |
| CONS |         • Less interpretable         | • More complex architecture |
|      | • Cannot specialize for rare classes |  • More models to maintain  |
|      |   • Error analysis more difficult    |      • Slower execution     |
|      |                                      |     • Error propagation     |
+------+--------------------------------------+-----------------------------+



DETAILED PERFORMANCE SUMMARY
------------------------------------------------------------------------------------------
+------------------+-----------------------+-----------------------+--------------------------+
|      Metric      | Chained Multi-outputs | Hierarchical Modeling |        Difference        |
+------------------+-----------------------+-----------------------+--------------------------+
|  Execution Time  |      5.78 seconds     |     13.80 seconds     | +-8.02 seconds (+-58.1%) |
| Average Accuracy |         77.18%        |         77.18%        |         +-0.00%          |
|   Model Count    |           28          |           28          |        +-0 models        |
+------------------+-----------------------+-----------------------+--------------------------+


RECOMMENDATION:
• The Hierarchical Modeling approach provides better interpretability despite being slower.
• Recommended for scenarios where understanding model decisions at each level is critical.
• Best for applications with clear hierarchical relationships between labels.


CONTEXT-SPECIFIC CONSIDERATIONS:
• Data volume: With larger datasets, the execution time difference will be more significant.
• Label relationships: If strong hierarchical dependencies exist, the hierarchical approach may be more appropriate.
• Maintenance resources: Consider the team's capacity to maintain multiple models vs. fewer more complex ones.
• Accuracy requirements: For critical applications, the approach with higher accuracy should be prioritized.