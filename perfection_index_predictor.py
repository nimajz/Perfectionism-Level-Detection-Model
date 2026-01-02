"""
Machine Learning Model for Predicting Player Perfection Index
This script reads event-level game data, aggregates it to session-level features,
and trains a model to predict a Perfection Index (0-1) for each player.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to data directory
DATA_DIR = Path("data/students-performance/raw data")

# Model selection: 'random_forest', 'gradient_boosting', or 'lightgbm'
MODEL_TYPE = 'gradient_boosting'

# Output file for predictions
OUTPUT_FILE = "perfection_index_predictions.csv"


# ============================================================================
# STEP 1: LOAD DATA (OPTIMIZED - CHUNKED PROCESSING)
# ============================================================================

def load_labels(data_dir):
    """
    Load all labels CSV files from the data directory.
    
    Returns:
        labels_df: Combined DataFrame of all labels data
    """
    print("Loading labels files...")
    
    data_dir = Path(data_dir)
    labels_files = sorted(data_dir.glob("*_labels.csv"))
    
    # Load all labels files
    labels_dfs = []
    for file in labels_files:
        print(f"  Loading {file.name}...")
        df = pd.read_csv(file, dtype={'session_id': 'str'})
        labels_dfs.append(df)
    
    labels_df = pd.concat(labels_dfs, ignore_index=True)
    # Convert session_id to string for consistent matching
    labels_df['session_id'] = labels_df['session_id'].astype(str)
    print(f"  Total label records: {len(labels_df)}")
    
    return labels_df


# ============================================================================
# STEP 2: AGGREGATE EVENT-LEVEL DATA TO SESSION-LEVEL FEATURES (OPTIMIZED)
# ============================================================================

def create_perfectionism_features_chunked(data_dir, labels_df, chunk_size=50000):
    """
    Aggregate event-level data to create session-level features using chunked processing.
    This version processes data in chunks to handle 30M+ rows efficiently without
    storing the full dataframe in RAM.
    
    Returns:
        features_df: DataFrame with session_id and all aggregated features
    """
    print("\nCreating session-level features (chunked processing)...")
    
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("*_train.csv"))
    
    # Required columns for feature engineering
    required_cols = ['session_id', 'elapsed_time', 'hover_duration', 'event_name', 
                     'level', 'room_fqid', 'fqid', 'fullscreen', 'hq', 'music']
    
    # Initialize incremental aggregation dictionaries
    # Each session_id maps to its accumulated feature values
    session_features = defaultdict(lambda: {
        # Time features - compute running statistics (memory efficient)
        'max_elapsed_time': 0.0,
        'hover_sum': 0.0,
        'hover_count': 0,
        'hover_max': 0.0,
        'level_elapsed': defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')}),
        
        # Retry/Revisit features - track counts
        'room_visits': defaultdict(int),
        'level_visits': [],  # List to track all level visits (including duplicates for revisit calculation)
        'fqid_visits': defaultdict(int),
        
        # Action features - count events
        'event_counts': defaultdict(int),
        'total_events': 0,
        
        # Settings features - track counts for mode calculation
        'fullscreen_counts': Counter(),
        'hq_counts': Counter(),
        'music_counts': Counter(),
        
        # Level progression
        'levels': set(),
        'rooms': set(),
    })
    
    # Process each file in chunks
    total_rows = 0
    for file_path in train_files:
        print(f"  Processing {file_path.name}...")
        
        try:
            # First, read just one row to check available columns
            sample_df = pd.read_csv(file_path, nrows=1)
            available_cols = set(sample_df.columns)
            
            # Select only columns that exist in the file
            cols_to_read = [col for col in required_cols if col in available_cols]
            # Always include session_id
            if 'session_id' not in cols_to_read:
                cols_to_read = ['session_id'] + cols_to_read
            
            # Process file in chunks
            chunk_iter = pd.read_csv(file_path, usecols=cols_to_read, chunksize=chunk_size, 
                                     dtype={'session_id': 'str'}, low_memory=False)
            
            for chunk_idx, chunk in enumerate(tqdm(chunk_iter, desc=f"    Chunks", leave=False)):
                # Convert numeric columns, handling errors
                if 'elapsed_time' in chunk.columns:
                    chunk['elapsed_time'] = pd.to_numeric(chunk['elapsed_time'], errors='coerce')
                if 'hover_duration' in chunk.columns:
                    chunk['hover_duration'] = pd.to_numeric(chunk['hover_duration'], errors='coerce')
                if 'level' in chunk.columns:
                    chunk['level'] = pd.to_numeric(chunk['level'], errors='coerce')
                
                # Group by session_id for efficient processing
                for session_id, group in chunk.groupby('session_id'):
                    feat = session_features[session_id]
                    
                    # Time features (compute running statistics for memory efficiency)
                    if 'elapsed_time' in group.columns:
                        valid_times = group['elapsed_time'].dropna()
                        if len(valid_times) > 0:
                            chunk_max = valid_times.max()
                            feat['max_elapsed_time'] = max(feat['max_elapsed_time'], chunk_max)
                            
                            # Track level-specific elapsed times
                            if 'level' in group.columns:
                                for level, level_group in group.groupby('level'):
                                    level_times = level_group['elapsed_time'].dropna()
                                    if len(level_times) > 0:
                                        level_min = level_times.min()
                                        level_max = level_times.max()
                                        feat['level_elapsed'][level]['min'] = min(
                                            feat['level_elapsed'][level]['min'], level_min
                                        )
                                        feat['level_elapsed'][level]['max'] = max(
                                            feat['level_elapsed'][level]['max'], level_max
                                        )
                    
                    if 'hover_duration' in group.columns:
                        valid_hovers = group['hover_duration'].dropna()
                        if len(valid_hovers) > 0:
                            feat['hover_sum'] += valid_hovers.sum()
                            feat['hover_count'] += len(valid_hovers)
                            feat['hover_max'] = max(feat['hover_max'], valid_hovers.max())
                    
                    # Retry/Revisit features
                    if 'room_fqid' in group.columns:
                        for room in group['room_fqid'].dropna():
                            feat['room_visits'][room] += 1
                            feat['rooms'].add(room)
                    
                    if 'level' in group.columns:
                        valid_levels = group['level'].dropna()
                        feat['level_visits'].extend(valid_levels.tolist())
                        feat['levels'].update(valid_levels.tolist())
                    
                    if 'fqid' in group.columns:
                        for fqid in group['fqid'].dropna():
                            feat['fqid_visits'][fqid] += 1
                    
                    # Action features
                    feat['total_events'] += len(group)
                    if 'event_name' in group.columns:
                        event_counts = group['event_name'].value_counts()
                        for event_name, count in event_counts.items():
                            feat['event_counts'][event_name] += count
                    
                    # Settings features (use Counter for memory efficiency)
                    if 'fullscreen' in group.columns:
                        valid_fullscreen = group['fullscreen'].dropna()
                        if len(valid_fullscreen) > 0:
                            feat['fullscreen_counts'].update(valid_fullscreen.tolist())
                    if 'hq' in group.columns:
                        valid_hq = group['hq'].dropna()
                        if len(valid_hq) > 0:
                            feat['hq_counts'].update(valid_hq.tolist())
                    if 'music' in group.columns:
                        valid_music = group['music'].dropna()
                        if len(valid_music) > 0:
                            feat['music_counts'].update(valid_music.tolist())
                
                total_rows += len(chunk)
        
        except Exception as e:
            print(f"    Error processing {file_path.name}: {e}")
            continue
    
    print(f"  Processed {total_rows:,} total rows")
    print(f"  Aggregating features for {len(session_features):,} sessions...")
    
    # Create a fast lookup dictionary for labels
    labels_dict = {}
    question_cols = [col for col in labels_df.columns 
                     if col not in ['session_id', 'save_code'] 
                     and pd.api.types.is_numeric_dtype(labels_df[col])]
    
    for _, row in labels_df.iterrows():
        session_id_str = str(row['session_id'])
        correct_answers = 0
        total_answers = 0
        
        for col in question_cols:
            val = row[col]
            if pd.notna(val):
                correct_answers += val
                total_answers += 1
        
        labels_dict[session_id_str] = {
            'accuracy': correct_answers / total_answers if total_answers > 0 else 0,
            'num_questions_attempted': total_answers,
            'num_questions_correct': correct_answers,
        }
    
    # Convert aggregated data to final feature DataFrame
    features_list = []
    
    for session_id, feat in tqdm(session_features.items(), desc="  Computing features"):
        # Time features (from running statistics)
        total_time = feat['max_elapsed_time']
        avg_hover_duration = feat['hover_sum'] / feat['hover_count'] if feat['hover_count'] > 0 else 0
        max_hover_duration = feat['hover_max']
        total_hover_time = feat['hover_sum']
        
        # Average time per level
        level_durations = []
        for level, time_range in feat['level_elapsed'].items():
            if time_range['min'] != float('inf') and time_range['max'] != float('-inf'):
                level_durations.append(time_range['max'] - time_range['min'])
        avg_time_per_level = np.mean(level_durations) if level_durations else 0
        
        # Retry/Revisit features
        room_visits = feat['room_visits']
        revisits = sum(1 for count in room_visits.values() if count > 1)
        avg_visits_per_room = np.mean(list(room_visits.values())) if room_visits else 0
        
        level_visits = feat['level_visits']
        level_revisits = len(level_visits) - len(set(level_visits)) if level_visits else 0
        
        fqid_visits = feat['fqid_visits']
        fqid_revisits = sum(1 for count in fqid_visits.values() if count > 1)
        avg_visits_per_fqid = np.mean(list(fqid_visits.values())) if fqid_visits else 0
        
        # Action features
        total_events = feat['total_events']
        event_counts = feat['event_counts']
        navigate_clicks = event_counts.get('navigate_click', 0)
        notebook_clicks = event_counts.get('notebook_click', 0)
        object_clicks = event_counts.get('object_click', 0)
        person_clicks = event_counts.get('person_click', 0)
        text_entered = event_counts.get('text_entered', 0)
        cutscene_clicks = event_counts.get('cutscene_click', 0)
        
        # Settings features (compute mode from Counter)
        fullscreen_counts = feat['fullscreen_counts']
        if fullscreen_counts:
            total_fullscreen = sum(fullscreen_counts.values())
            fullscreen_mode = fullscreen_counts.most_common(1)[0][0]
            fullscreen_ratio = fullscreen_counts[fullscreen_mode] / total_fullscreen
        else:
            fullscreen_mode = 0
            fullscreen_ratio = 0
        
        hq_counts = feat['hq_counts']
        if hq_counts:
            total_hq = sum(hq_counts.values())
            hq_mode = hq_counts.most_common(1)[0][0]
            hq_ratio = hq_counts[hq_mode] / total_hq
        else:
            hq_mode = 0
            hq_ratio = 0
        
        music_counts = feat['music_counts']
        if music_counts:
            total_music = sum(music_counts.values())
            music_mode = music_counts.most_common(1)[0][0]
            music_ratio = music_counts[music_mode] / total_music
        else:
            music_mode = 0
            music_ratio = 0
        
        # Level progression
        unique_levels = len(feat['levels'])
        max_level = max(feat['levels']) if feat['levels'] else 0
        unique_rooms = len(feat['rooms'])
        
        # Accuracy features (from labels - use fast dictionary lookup)
        session_id_str = str(session_id)
        label_data = labels_dict.get(session_id_str, {
            'accuracy': 0,
            'num_questions_attempted': 0,
            'num_questions_correct': 0,
        })
        accuracy = label_data['accuracy']
        num_questions_attempted = label_data['num_questions_attempted']
        num_questions_correct = label_data['num_questions_correct']
        
        # Create feature dictionary
        features = {
            'session_id': session_id,
            
            # Time features
            'total_time': total_time,
            'avg_hover_duration': avg_hover_duration,
            'max_hover_duration': max_hover_duration,
            'total_hover_time': total_hover_time,
            'avg_time_per_level': avg_time_per_level,
            
            # Retry/Revisit features
            'room_revisits': revisits,
            'avg_visits_per_room': avg_visits_per_room,
            'level_revisits': level_revisits,
            'fqid_revisits': fqid_revisits,
            'avg_visits_per_fqid': avg_visits_per_fqid,
            
            # Action features
            'total_events': total_events,
            'navigate_clicks': navigate_clicks,
            'notebook_clicks': notebook_clicks,
            'object_clicks': object_clicks,
            'person_clicks': person_clicks,
            'text_entered': text_entered,
            'cutscene_clicks': cutscene_clicks,
            
            # Settings features
            'fullscreen_mode': fullscreen_mode,
            'fullscreen_ratio': fullscreen_ratio,
            'hq_mode': hq_mode,
            'hq_ratio': hq_ratio,
            'music_mode': music_mode,
            'music_ratio': music_ratio,
            
            # Level progression
            'unique_levels': unique_levels,
            'max_level': max_level,
            'unique_rooms': unique_rooms,
            
            # Accuracy features
            'accuracy': accuracy,
            'num_questions_attempted': num_questions_attempted,
            'num_questions_correct': num_questions_correct,
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"  Created features for {len(features_df)} sessions")
    
    return features_df


# ============================================================================
# STEP 3: CREATE TARGET VARIABLE (Perfection Index)
# ============================================================================

def create_perfection_index_target(features_df):
    """
    Create a heuristic-based Perfection Index target (0-1) for training.
    
    Since we don't have ground truth labels for perfection index, we create
    a composite score based on perfectionism indicators:
    - High accuracy (weight: 0.4)
    - High hover duration relative to total time (weight: 0.2)
    - Moderate retries/revisits (weight: 0.2) - too many might indicate confusion
    - Consistent quality settings (weight: 0.1)
    - Thorough exploration (weight: 0.1)
    
    NOTE: Replace this function with your actual labels if you have them!
    """
    print("\nCreating heuristic-based Perfection Index target...")
    
    # Normalize features to 0-1 scale for combining
    def normalize(series):
        if series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())
    
    # Accuracy component (0.4 weight)
    accuracy_score = features_df['accuracy'].fillna(0)
    
    # Hover duration component (0.2 weight) - time spent examining elements
    hover_score = normalize(features_df['avg_hover_duration'].fillna(0))
    
    # Retry component (0.2 weight) - moderate retries indicate perfectionism
    # Too many retries might indicate confusion, so we use a sigmoid-like transformation
    retry_score = normalize(
        np.clip(features_df['room_revisits'].fillna(0) + features_df['fqid_revisits'].fillna(0), 0, 10) / 10
    )
    
    # Settings consistency (0.1 weight) - preference for quality
    settings_score = (
        normalize(features_df['hq_ratio'].fillna(0)) * 0.5 +
        normalize(features_df['fullscreen_ratio'].fillna(0)) * 0.3 +
        normalize(features_df['music_ratio'].fillna(0)) * 0.2
    )
    
    # Exploration thoroughness (0.1 weight)
    exploration_score = normalize(features_df['unique_rooms'].fillna(0))
    
    # Combine all components
    perfection_index = (
        accuracy_score * 0.4 +
        hover_score * 0.2 +
        retry_score * 0.2 +
        settings_score * 0.1 +
        exploration_score * 0.1
    )
    
    # Ensure values are between 0 and 1
    perfection_index = np.clip(perfection_index, 0, 1)
    
    print(f"  Perfection Index range: {perfection_index.min():.3f} - {perfection_index.max():.3f}")
    print(f"  Mean Perfection Index: {perfection_index.mean():.3f}")
    
    return perfection_index


# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, model_type='gradient_boosting'):
    """
    Train a regression model to predict Perfection Index.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: Type of model to use
    
    Returns:
        model: Trained model
        scaler: Fitted scaler (if needed)
    """
    print(f"\nTraining {model_type} model...")
    
    # Feature scaling (not always needed for tree-based models, but helps with some)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model based on type
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)  # Tree models don't need scaling
        scaler = None
        
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)  # Tree models don't need scaling
        scaler = None
        
    elif model_type == 'lightgbm':
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            scaler = None
        except ImportError:
            print("  LightGBM not available, falling back to GradientBoosting...")
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            scaler = None
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate on validation set
    if scaler:
        y_pred = model.predict(X_val_scaled)
    else:
        y_pred = model.predict(X_val)
    
    y_pred = np.clip(y_pred, 0, 1)  # Ensure predictions are in [0, 1]
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"  Validation MSE: {mse:.4f}")
    print(f"  Validation MAE: {mae:.4f}")
    print(f"  Validation R²: {r2:.4f}")
    
    return model, scaler


# ============================================================================
# STEP 5: PREDICT AND SAVE RESULTS
# ============================================================================

def predict_and_save(model, scaler, X_all, session_ids, output_file):
    """
    Make predictions for all sessions and save to CSV.
    
    Args:
        model: Trained model
        scaler: Fitted scaler (if used)
        X_all: Feature matrix for all sessions
        session_ids: Session IDs corresponding to X_all
        output_file: Path to output CSV file
    """
    print(f"\nMaking predictions...")
    
    # Make predictions
    if scaler:
        X_scaled = scaler.transform(X_all)
        predictions = model.predict(X_scaled)
    else:
        predictions = model.predict(X_all)
    
    # Ensure predictions are in [0, 1] range
    predictions = np.clip(predictions, 0, 1)
    
    # Create output DataFrame
    results_df = pd.DataFrame({
        'session_id': session_ids,
        'perfection_index': predictions
    })
    
    # Sort by session_id for consistency
    results_df = results_df.sort_values('session_id').reset_index(drop=True)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"  Predictions saved to {output_file}")
    print(f"  Perfection Index range: {predictions.min():.3f} - {predictions.max():.3f}")
    print(f"  Mean Perfection Index: {predictions.mean():.3f}")
    
    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function that orchestrates the entire pipeline."""
    
    print("=" * 70)
    print("PERFECTION INDEX PREDICTOR")
    print("=" * 70)
    
    # Step 1: Load labels (labels are small, so we can load them normally)
    labels_df = load_labels(DATA_DIR)
    
    # Step 2: Create features using chunked processing (memory efficient)
    features_df = create_perfectionism_features_chunked(DATA_DIR, labels_df, chunk_size=50000)
    
    # Step 3: Create target variable
    features_df['perfection_index_target'] = create_perfection_index_target(features_df)
    
    # Step 4: Prepare data for modeling
    print("\nPreparing data for modeling...")
    
    # Select feature columns (exclude session_id and target)
    feature_columns = [col for col in features_df.columns 
                      if col not in ['session_id', 'perfection_index_target']]
    
    X = features_df[feature_columns].fillna(0)  # Fill missing values with 0
    y = features_df['perfection_index_target']
    session_ids = features_df['session_id']
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features used: {len(feature_columns)}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Step 5: Train model
    model, scaler = train_model(X_train, y_train, X_val, y_val, MODEL_TYPE)
    
    # Step 6: Make predictions on all data
    results_df = predict_and_save(model, scaler, X, session_ids, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review predictions in {OUTPUT_FILE}")
    print(f"  2. If you have actual Perfection Index labels, replace the")
    print(f"     create_perfection_index_target() function to use your labels")
    print(f"  3. Adjust feature engineering in create_perfectionism_features()")
    print(f"     if needed")
    print(f"  4. Tune hyperparameters for better performance")
    print("\n")


if __name__ == "__main__":
    main()

