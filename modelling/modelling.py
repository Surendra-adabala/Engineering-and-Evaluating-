from model.randomforest import RandomForest
from Config import Config
import time
import numpy as np


def model_predict(data, df, name):
    """Standard single target modeling"""
    if data.X_train is None:
        print("Skipping due to insufficient data")
        return None
        
    start_time = time.time()
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.get_X_test())
    model.print_results(data)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    return model


def model_evaluate(model, data):
    """Evaluate a trained model"""
    if model is None or data.X_train is None:
        return
    model.print_results(data)


def chained_model_predict(data, df, name):
    """Chained multi-output modeling approach (Design Decision 1)
    
    This approach trains a single model for each combination of target variables:
    - Type 2
    - Type 2 + Type 3
    - Type 2 + Type 3 + Type 4
    """
    if data.X_train is None:
        print("Skipping chained modeling due to insufficient data")
        return None
    
    start_time = time.time()
    print("Chained RandomForest")
    
    # Use fewer estimators for faster training in this complex model
    model = RandomForest(
        f"ChainedRandomForest_{name}", 
        data.get_embeddings(), 
        data.get_type(), 
        mode="chained",
        n_estimators=300
    )
    
    model.train(data)
    model.predict(data.get_X_test())
    model.print_results(data)
    
    # Save model for later use
    model.save_model()
    
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return model


def hierarchical_model_predict(data, df, name):
    """Hierarchical modeling approach (Design Decision 2)
    
    This approach creates a tree of models:
    - Level 1: Models for Type 2
    - Level 2: For each Type 2 class, models for Type 3
    - Level 3: For each Type 3 class, models for Type 4
    """
    if data.X_train is None:
        print("Skipping hierarchical base model due to insufficient data")
        return None
    
    start_time = time.time()
    model_creation_time = 0
    
    # Train base model for Type 2
    print(f"Hierarchical RandomForest - Base Model for {data.get_target_column()}")
    base_model = RandomForest(
        f"HierarchicalRandomForest_{name}_{data.get_target_column()}", 
        data.get_embeddings(), 
        data.get_type(), 
        mode="hierarchical"
    )
    
    # Use fewer estimators for child models
    child_estimators = 200
    
    base_model.train(data)
    base_model.predict(data.get_X_test())
    base_model.print_results(data)
    
    # If this is Type 2, we need to create child models for each Type 2 class
    if data.get_target_column() == Config.TYPE_COLS[0] and hasattr(data, 'classes'):
        original_df = data.get_original_df()
        embeddings = data.get_embeddings()
        
        # Track all models for potential ensemble predictions
        all_models = {data.get_target_column(): base_model}
        
        # For each unique class in Type 2
        for class_value in data.classes:
            # Skip if class value is None or empty
            if class_value is None or (isinstance(class_value, str) and not class_value.strip()):
                continue
                
            # Create filter condition for this class
            filter_condition = {Config.TYPE_COLS[0]: class_value}
            print(f"\nTraining Type 3 model for {Config.TYPE_COLS[0]}={class_value}")
            
            child_start_time = time.time()
            
            # Create new data object filtered for this class
            type3_data = data.__class__(
                embeddings,
                original_df,
                target_col=Config.TYPE_COLS[1],
                filter_condition=filter_condition
            )
            
            if type3_data.X_train is not None and hasattr(type3_data, 'classes') and len(type3_data.classes) > 0:
                # Train Type 3 model for this Type 2 class
                type3_model = RandomForest(
                    f"Type3Model_{class_value}", 
                    type3_data.get_embeddings(), 
                    type3_data.get_type(), 
                    mode="hierarchical",
                    n_estimators=child_estimators
                )
                
                type3_model.train(type3_data)
                type3_model.predict(type3_data.get_X_test())
                type3_model.print_results(type3_data)
                
                # Add as child model to base model
                base_model.add_child_model(type3_model)
                
                # Store in all_models dictionary
                model_key = f"{Config.TYPE_COLS[0]}={class_value}_{Config.TYPE_COLS[1]}"
                all_models[model_key] = type3_model
                
                # For each Type 3 class, create Type 4 models if needed
                if hasattr(type3_data, 'classes') and type3_data.classes:
                    for type3_class in type3_data.classes:
                        # Skip if class value is None or empty
                        if type3_class is None or (isinstance(type3_class, str) and not type3_class.strip()):
                            continue
                            
                        # Create nested filter condition
                        nested_filter = {
                            Config.TYPE_COLS[0]: class_value,
                            Config.TYPE_COLS[1]: type3_class
                        }
                        
                        print(f"\nTraining Type 4 model for {Config.TYPE_COLS[0]}={class_value}, {Config.TYPE_COLS[1]}={type3_class}")
                        
                        # Create new data object with nested filter
                        type4_data = data.__class__(
                            embeddings,
                            original_df,
                            target_col=Config.TYPE_COLS[2],
                            filter_condition=nested_filter
                        )
                        
                        if type4_data.X_train is not None and hasattr(type4_data, 'classes') and len(type4_data.classes) > 0:
                            # Train Type 4 model
                            type4_model = RandomForest(
                                f"Type4Model_{class_value}_{type3_class}", 
                                type4_data.get_embeddings(), 
                                type4_data.get_type(), 
                                mode="hierarchical",
                                n_estimators=child_estimators
                            )
                            
                            type4_model.train(type4_data)
                            type4_model.predict(type4_data.get_X_test())
                            type4_model.print_results(type4_data)
                            
                            # Add as child model to Type 3 model
                            type3_model.add_child_model(type4_model)
                            
                            # Store in all_models dictionary
                            model_key = f"{Config.TYPE_COLS[0]}={class_value}_{Config.TYPE_COLS[1]}={type3_class}_{Config.TYPE_COLS[2]}"
                            all_models[model_key] = type4_model
                            
            model_creation_time += time.time() - child_start_time
            
        # Save base model (which contains references to all child models)
        base_model.save_model()
        
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    return base_model