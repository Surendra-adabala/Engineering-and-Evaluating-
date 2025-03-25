class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'
    
    # Multi-label classification settings
    CHAINED_MODE = False  # When True, use Design Decision 1, else use Design Decision 2
    MIN_CLASS_SAMPLES = 3  # Minimum samples per class
    RANDOM_SEED = 0  # For reproducibility
    
    # Pipeline configuration
    PIPELINE_MODE = "auto"  # "chained", "hierarchical", or "auto" (uses CHAINED_MODE)
    
    # Paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    MODELS_DIR = "saved_models"
    LOGS_DIR = "logs"
    
    # Visualization settings
    VISUALIZE_RESULTS = True
    GENERATE_REPORTS = True
    
    # RandomForest parameters
    RF_PARAMS = {
        "standard": {
            "n_estimators": 500,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": RANDOM_SEED
        },
        "chained": {
            "n_estimators": 300,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": RANDOM_SEED
        },
        "hierarchical": {
            "base": {
                "n_estimators": 500,
                "class_weight": "balanced_subsample",
                "max_features": "sqrt",
                "n_jobs": -1,
                "random_state": RANDOM_SEED
            },
            "child": {
                "n_estimators": 200,
                "class_weight": "balanced_subsample",
                "max_features": "sqrt",
                "n_jobs": -1,
                "random_state": RANDOM_SEED
            }
        }
    }
    
    # TF-IDF parameters
    TFIDF_PARAMS = {
        "max_features": 2000,
        "min_df": 4,
        "max_df": 0.9
    } 