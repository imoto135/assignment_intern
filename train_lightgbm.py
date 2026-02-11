"""
Baseline model training script using LightGBM for ETTh1 dataset.
This script prepares the data, creates lag features, and sets up a LightGBM model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime
import wandb

# Parameters
DATA_DIR = Path('data/split')
OUTPUT_DIR = Path('experiments/baseline')
DATASET = 'ETTh1'
TARGET_COL = 'OT'
FEATURE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# Lag feature parameters
LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

# LightGBM parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


def load_data():
    """Load train, validation, and test datasets."""
    print(f"Loading {DATASET} dataset...")
    
    train_path = DATA_DIR / f'{DATASET}_train.csv'
    val_path = DATA_DIR / f'{DATASET}_val.csv'
    test_path = DATA_DIR / f'{DATASET}_test.csv'
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Parse date column
    for df in [train_df, val_df, test_df]:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def create_lag_features(df, feature_cols, lag_steps):
    """Create lag features for each feature column."""
    df_lag = df.copy()
    
    for col in feature_cols:
        for lag in lag_steps:
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
    
    return df_lag


def create_rolling_features(df, feature_cols, windows):
    """Create rolling mean features for each feature column."""
    df_rolling = df.copy()
    
    for col in feature_cols:
        for window in windows:
            df_rolling[f'{col}_rolling_mean_{window}'] = (
                df_rolling[col].shift(1).rolling(window=window).mean()
            )
            df_rolling[f'{col}_rolling_std_{window}'] = (
                df_rolling[col].shift(1).rolling(window=window).std()
            )
    
    return df_rolling


def add_time_features(df):
    """Extract time-based features from date column."""
    df_time = df.copy()
    
    df_time['hour'] = df_time['date'].dt.hour
    df_time['day_of_week'] = df_time['date'].dt.dayofweek
    df_time['day_of_month'] = df_time['date'].dt.day
    df_time['month'] = df_time['date'].dt.month
    
    # Cyclical encoding
    df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
    df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
    
    return df_time


def prepare_features(train_df, val_df, test_df):
    """Prepare all features for training."""
    print("\nCreating lag features...")
    train_df = create_lag_features(train_df, FEATURE_COLS, LAG_STEPS)
    val_df = create_lag_features(val_df, FEATURE_COLS, LAG_STEPS)
    test_df = create_lag_features(test_df, FEATURE_COLS, LAG_STEPS)
    
    print("Creating rolling features...")
    train_df = create_rolling_features(train_df, FEATURE_COLS, ROLLING_WINDOWS)
    val_df = create_rolling_features(val_df, FEATURE_COLS, ROLLING_WINDOWS)
    test_df = create_rolling_features(test_df, FEATURE_COLS, ROLLING_WINDOWS)
    
    print("Adding time features...")
    train_df = add_time_features(train_df)
    val_df = add_time_features(val_df)
    test_df = add_time_features(test_df)
    
    # Drop rows with NaN values (caused by lag features)
    print("\nDropping rows with NaN values...")
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    
    print(f"After preprocessing:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def prepare_datasets(train_df, val_df, test_df):
    """Prepare X, y for training."""
    # Exclude date and target from features
    feature_columns = [col for col in train_df.columns 
                      if col not in ['date', TARGET_COL]]
    
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[feature_columns]
    y_val = val_df[TARGET_COL]
    
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COL]
    
    print(f"\nFeature count: {len(feature_columns)}")
    print(f"Feature names: {feature_columns[:10]}... (showing first 10)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns


def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    print("\n" + "="*60)
    print("Training LightGBM Model")
    print("="*60)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    print("\nStarting training...")
    print(f"Parameters: {json.dumps(LGBM_PARAMS, indent=2)}")
    
    # Custom callback to log to wandb
    def log_to_wandb(env):
        if env.evaluation_result_list:
            for data_name, eval_name, result, is_higher_better in env.evaluation_result_list:
                metric_name = f"{data_name}_{eval_name}"
                wandb.log({metric_name: result}, step=env.iteration)
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=10),
            lgb.callback.early_stopping(EARLY_STOPPING_ROUNDS),
            log_to_wandb
        ]
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")
    
    return model


def evaluate_model(model, X, y, dataset_name='Test'):
    """Evaluate model performance."""
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'predictions': y_pred,
        'actuals': y
    }


def save_model_and_results(model, metrics, feature_columns):
    """Save trained model and results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = OUTPUT_DIR / f'lgbm_{DATASET}_{timestamp}.txt'
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics and configuration
    results = {
        'dataset': DATASET,
        'timestamp': timestamp,
        'model_type': 'LightGBM',
        'parameters': LGBM_PARAMS,
        'num_boost_round': NUM_BOOST_ROUND,
        'best_iteration': model.best_iteration,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'lag_steps': LAG_STEPS,
        'rolling_windows': ROLLING_WINDOWS,
        'num_features': len(feature_columns),
        'metrics': {
            'train': {k: float(v) for k, v in metrics['train'].items() if k not in ['predictions', 'actuals']},
            'val': {k: float(v) for k, v in metrics['val'].items() if k not in ['predictions', 'actuals']},
            'test': {k: float(v) for k, v in metrics['test'].items() if k not in ['predictions', 'actuals']}
        }
    }
    
    results_path = OUTPUT_DIR / f'results_{DATASET}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Log final metrics to wandb
    wandb.log({
        'final_train_rmse': metrics['train']['rmse'],
        'final_train_mae': metrics['train']['mae'],
        'final_val_rmse': metrics['val']['rmse'],
        'final_val_mae': metrics['val']['mae'],
        'final_test_rmse': metrics['test']['rmse'],
        'final_test_mae': metrics['test']['mae'],
        'best_iteration': model.best_iteration,
        'num_features': len(feature_columns)
    })
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance_path = OUTPUT_DIR / f'feature_importance_{DATASET}_{timestamp}.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Log feature importance to wandb
    importance_table = wandb.Table(data=importance_df.head(20).values.tolist(),
                                   columns=['feature', 'importance'])
    wandb.log({'top_features': importance_table})
    
    # Save model artifact
    wandb.save(str(model_path))
    
    # Display top 20 features
    print("\nTop 20 Important Features:")
    print(importance_df.head(20).to_string(index=False))


def main():
    """Main training pipeline."""
    # Initialize wandb
    run_name = f"lightgbm_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project="timeseries-forecast",
        name=run_name,
        config={
            'dataset': DATASET,
            'model': 'LightGBM',
            'lgbm_params': LGBM_PARAMS,
            'num_boost_round': NUM_BOOST_ROUND,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'lag_steps': LAG_STEPS,
            'rolling_windows': ROLLING_WINDOWS
        },
        tags=['baseline', 'lightgbm', DATASET]  # タグで検索しやすく
    )
    
    print("="*60)
    print("LightGBM Baseline Training")
    print(f"Dataset: {DATASET}")
    print(f"wandb Project: timeseries-forecast")
    print(f"wandb Run: {run_name}")
    print("="*60)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare features
    train_df, val_df, test_df = prepare_features(train_df, val_df, test_df)
    
    # Prepare datasets
    X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = prepare_datasets(
        train_df, val_df, test_df
    )
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on all datasets
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    metrics = {}
    metrics['train'] = evaluate_model(model, X_train, y_train, 'Train')
    metrics['val'] = evaluate_model(model, X_val, y_val, 'Validation')
    metrics['test'] = evaluate_model(model, X_test, y_test, 'Test')
    
    # Save model and results
    save_model_and_results(model, metrics, feature_columns)
    
    # Finish wandb run
    wandb.finish()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
