# Email Classification System

## Overview
This system provides an end-to-end solution for hierarchical email classification using machine learning approaches. The system processes emails from AppGallery and In-App Purchase domains, classifying them into multiple hierarchical categories (Type 2, Type 3, and Type 4 classification levels).

The project implements and compares two distinct modeling approaches:
1. **Chained Multi-output Approach**: Single models that predict multiple targets simultaneously
2. **Hierarchical Modeling Approach**: Multiple specialized models organized in a hierarchical structure

## Project Structure

```
Email_Classification_System/
├── data/                       # Contains email datasets  
│   ├── AppGallery.csv          # Emails related to AppGallery
│   └── Purchasing.csv          # Emails related to In-App Purchases
├── modelling/                  # Core ML pipeline components
│   ├── base_pipeline.py        # Base pipeline architecture
│   ├── data_wrapper.py         # Data handling and preprocessing
│   ├── evaluation_pipeline.py  # Metrics calculation and evaluation
│   ├── modeling_pipeline.py    # Model training pipeline stages
│   └── preprocessing_pipeline.py # Data preprocessing pipeline stages
├── model/                      # Model implementations
│   └── randomforest.py         # RandomForest classifier implementation
├── utils/                      # Utility functions
│   ├── __init__.py             # Package initialization
│   ├── logging_utils.py        # Logging configuration utilities
│   ├── metrics_utils.py        # Performance metrics calculation
│   └── visualization_utils.py  # Visualization and chart creation
├── logs/                       # Log files directory
├── saved_models/               # Directory for saved model files
├── results/                    # Results directory
│   ├── metrics.json            # Performance metrics
│   ├── comparison_report.md    # Comparison report
│   └── visualizations/         # Chart visualizations
├── comparison/                 # Comparison results
│   ├── chained/                # Results for chained approach
│   ├── hierarchical/           # Results for hierarchical approach
│   ├── design_comparison.md    # Overall comparison report
│   └── visualizations/         # Comparison visualizations
├── Config.py                   # Configuration settings
├── main.py                     # Main entry point for individual runs
├── run_comparison.py           # Script to run and compare both approaches
├── preprocess.py               # Text preprocessing functions
├── embeddings.py               # TF-IDF embedding generation
└── README.md                   # This file
```

## Design Approaches

### 1. Chained Multi-output Approach

**Overview:**
- A single model per group that predicts multiple target classes simultaneously
- Uses a chained prediction pattern where predictions flow from Type 2 → Type 3 → Type 4
- Fewer models to maintain (simpler architecture)

**Implementation:**
- Creates a single RandomForest model that predicts combined target values
- Uses concatenated target values (e.g., "Problem/Fault+Payment+Risk Control")
- Faster execution time but less interpretable

### 2. Hierarchical Modeling Approach

**Overview:**
- Multiple specialized models organized in a hierarchical tree structure
- Separate models for each level and class combination
- More models to maintain but better interpretability

**Implementation:**
- Creates base models for Type 2 classification
- For each Type 2 class, creates child models for Type 3 classification
- For each Type 3 class, creates child models for Type 4 classification
- Models are linked in a parent-child relationship
- Slower execution but more interpretable and flexible

## Data Structure

The email data is classified in a hierarchical manner:
- **Type 1**: Domain category (AppGallery & Games, In-App Purchase)
- **Type 2**: Main issue type (Problem/Fault, Suggestion, Others)
- **Type 3**: Specific issue area (varies by Type 2)
- **Type 4**: Detailed issue type (varies by Type 3)

## Results and Comparison

### Performance Metrics

| Metric              | Chained Multi-outputs | Hierarchical Modeling | Difference               |
|---------------------|----------------------|----------------------|--------------------------|
| Execution Time      | 7.08 seconds         | 18.76 seconds        | +11.68 seconds (+62.2%)  |
| Average Accuracy    | 77.18%               | 82.87%               | +5.70%                   |
| Average F1 Weighted | 75.30%               | 81.61%               | +6.31%                   |
| Model Count         | 64                   | 54                   | -10 models               |

### Key Findings

**Chained Approach Advantages:**
- Simpler architecture
- Fewer models to train
- Faster execution
- Easier maintenance

**Hierarchical Approach Advantages:**
- Better interpretability
- Class-specific models
- Clearer error tracing
- Flexible structure
- Higher accuracy and F1 scores

**Recommendation:**
The Hierarchical Modeling approach provides better accuracy and interpretability despite being slower. It is recommended for scenarios where understanding model decisions at each level is critical and for applications with clear hierarchical relationships between labels.

## File Descriptions

### Core Files
- `Config.py`: Configuration settings for the entire pipeline including column names, file paths, and model parameters
- `main.py`: Main entry point for running individual approaches (chained or hierarchical)
- `run_comparison.py`: Script for running and comparing both approaches, generating comprehensive reports
- `preprocess.py`: Contains text preprocessing functions including cleaning, noise removal, and deduplication
- `embeddings.py`: Handles TF-IDF embedding generation for converting text to numerical features

### Utility Modules
- `utils/logging_utils.py`: Configures logging for the entire project
- `utils/metrics_utils.py`: Calculates performance metrics like accuracy, precision, recall, and F1 scores
- `utils/visualization_utils.py`: Creates visualizations for model performance and comparison

## Installation and Setup

### Prerequisites
- Python 3.7+
- scikit-learn
- pandas
- numpy
- prettytable
- matplotlib

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Email_Classification_System
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install scikit-learn pandas numpy prettytable matplotlib
```

## Usage Instructions

### Running the Comparison
To compare both approaches (Chained and Hierarchical):

```bash
python run_comparison.py
```

This will:
1. Run both modeling approaches
2. Generate detailed metrics
3. Create visualizations
4. Produce a comparison report

### Command Line Arguments

The comparison script supports the following arguments:

```bash
python run_comparison.py [--skip-chained] [--skip-hierarchical] [--results-dir DIRECTORY]
```

- `--skip-chained`: Skip running the chained approach
- `--skip-hierarchical`: Skip running the hierarchical approach
- `--results-dir`: Specify the directory to save results (default: ./comparison)

To save the output to a file:

```bash
python run_comparison.py > detailed_classification_report.txt
```

### Running Individual Approaches

To run a specific approach:

```bash
python main.py --mode [chained|hierarchical] --results-dir ./results --visualize --report
```

## Pipeline Architecture

The system uses a modular pipeline architecture with distinct stages:

1. **Data Loading Stage**: Loads data from CSV files
2. **Preprocessing Stage**: Cleans and prepares the data (uses `preprocess.py`)
3. **Embeddings Stage**: Converts text to numerical features (uses `embeddings.py`)
4. **Data Preparation Stage**: Creates appropriate data splits
5. **Model Training Stage**: Trains models based on the selected approach
6. **Evaluation Stage**: Calculates performance metrics
7. **Reporting Stage**: Generates reports and visualizations

## Code Explanation

### Key Components

#### 1. Data Wrapper (modelling/data_wrapper.py)
Handles data preprocessing, filtering, and train/test splitting. Implements:
- Filtering by condition for hierarchical modeling
- Minimum class samples thresholding
- Different data modes (standard, chained, hierarchical)

#### 2. Pipeline Stages (modelling/*.py)
Each stage in the pipeline follows a consistent interface:
- Requires specific inputs from the context
- Processes the data
- Provides outputs to the context

#### 3. Model Implementation (model/randomforest.py)
RandomForest classifier implementation with:
- Support for both chained and hierarchical modes
- Parent-child relationship for hierarchical models
- Model saving and loading capabilities

#### 4. Evaluation Pipeline (modelling/evaluation_pipeline.py)
Calculates metrics for model evaluation:
- Accuracy, precision, recall, F1 scores
- Group-wise and class-wise metrics
- Hierarchical evaluation for all model levels

#### 5. Utility Functions (utils/)
- Logging configuration and management
- Metrics calculation and reporting
- Visualization generation for comparing approaches

## Handling Data Sparsity

The system includes special handling for data sparsity, particularly in hierarchical modeling:
- Custom minimum class sample thresholds based on modeling mode
- For hierarchical mode with filtered data, threshold is reduced to 2 
- Robust evaluation with fallback mechanisms
- Zero-division handling in metric calculations

## Logging

The system uses a comprehensive logging framework:
- Log files are stored in the `logs` directory
- Different logging levels for debugging and production
- Detailed logging of each pipeline stage execution
- Performance metrics and timing information

## Future Improvements

1. **Network Visualization**: Add network visualization for hierarchical models (currently shows placeholder due to missing networkx module)
2. **Data Augmentation**: Implement techniques to address class imbalance and data sparsity
3. **Model Optimization**: Hyperparameter tuning for better performance
4. **Additional Models**: Support for more model types beyond RandomForest
5. **Interactive Visualization**: Add interactive dashboards for result exploration

## Contact

For any questions or issues, please contact [your-email@example.com]. 