"""
CatBoost model training script for time series forecasting on ETTh1 dataset.
Comparable with LightGBM baseline.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime
import wandb
import warnings
warnings.filterwarnings('ignore')

# Parameters
DATA_DIR = Path('data/split')
OUTPUT_DIR = Path('experiments/catboost')
DATASET = 'ETTh1'
TARGET_COL = 'OT'
FEATURE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# Feature engineering parameters (LightGBM と同じ)
LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

# CatBoost parameters
CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1
}

NUM_ITERATIONS = 500
EARLY_STOPPING_ROUNDS = 50


def load_data():
    """Load train, val, test data."""
    train_df = pd.read_csv(DATA_DIR / f'{DATASET}_train.csv')
    val_df = pd.read_csv(DATA_DIR / f'{DATASET}_val.csv')
    test_df = pd.read_csv(DATA_DIR / f'{DATASET}_test.csv')
    return train_df, val_df, test_df


def create_lag_features(df, feature_cols, lag_steps):
    """Create lag features (same as LightGBM)."""
    df_lag = df.copy()
    
    for col in feature_cols:
        for lag in lag_steps:
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
    
    return df_lag


def create_rolling_features(df, feature_cols, windows):
    """Create rolling statistics features (same as LightGBM)."""
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


def create_time_features(df):
    """Create time features (same as LightGBM)."""
    df_time = df.copy()
    
    # Extract time components
    df_time['hour'] = pd.to_datetime(df_time.index).hour
    df_time['day_of_week'] = pd.to_datetime(df_time.index).dayofweek
    df_time['day_of_month'] = pd.to_datetime(df_time.index).day
    df_time['month'] = pd.to_datetime(df_time.index).month
    
    # Cyclic encoding
    df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
    
    df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
    df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
    
    df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
    
    return df_time


def prepare_features(train_df, val_df, test_df):
    """Prepare all features (lag, rolling, time)."""
    print("\nPreparing features...")
    
    # Create lag features
    train_df = create_lag_features(train_df, FEATURE_COLS, LAG_STEPS)
    val_df = create_lag_features(val_df, FEATURE_COLS, LAG_STEPS)
    test_df = create_lag_features(test_df, FEATURE_COLS, LAG_STEPS)
    
    # Create rolling features
    train_df = create_rolling_features(train_df, FEATURE_COLS, ROLLING_WINDOWS)
    val_df = create_rolling_features(val_df, FEATURE_COLS, ROLLING_WINDOWS)
    test_df = create_rolling_features(test_df, FEATURE_COLS, ROLLING_WINDOWS)
    
    # Create time features
    train_df = create_time_features(train_df)
    val_df = create_time_features(val_df)
    test_df = create_time_features(test_df)
    
    # Drop rows with NaN (caused by lag and rolling features)
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    
    print(f"  Train set: {len(train_df)} samples")
    print(f"  Val set: {len(val_df)} samples")
    print(f"  Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def prepare_model_data(train_df, val_df, test_df):
    """Prepare data for CatBoost model."""
    print("\nPreparing model data...")
    
    # Separate features and target
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # Select only numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    feature_columns = list(X_train.columns)
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    print(f"  Features: {len(feature_columns)}")
    print(f"  Train samples: {len(X_train_scaled)}")
    print(f"  Val samples: {len(X_val_scaled)}")
    print(f"  Test samples: {len(X_test_scaled)}")
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled, scaler_y, feature_columns)


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    print("\n" + "="*60)
    print("Training CatBoost Model")
    print("="*60)
    
    print(f"\nCatBoost Parameters:")
    for key, value in CATBOOST_PARAMS.items():
        print(f"  {key}: {value}")
    
    # Create Pool
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    
    # Training with early stopping
    print(f"\nStarting training...")
    
    model = CatBoostRegressor(
        **CATBOOST_PARAMS,
        iterations=NUM_ITERATIONS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=50
    )
    
    best_epoch = model.best_iteration_
    print(f"\nBest epoch: {best_epoch + 1}")
    
    return model


def predict(model, X):
    """Make predictions."""
    return model.predict(X)


def evaluate_model(model, X, y, scaler_y, dataset_name='Test'):
    """Evaluate model performance."""
    print(f"\n{dataset_name} Evaluation:")
    print("-" * 40)
    
    y_pred = predict(model, X)
    
    # Inverse scale
    y = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
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
    model_path = OUTPUT_DIR / f'catboost_{DATASET}_{timestamp}.cbm'
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics and configuration
    results = {
        'dataset': DATASET,
        'timestamp': timestamp,
        'model_type': 'CatBoost',
        'parameters': CATBOOST_PARAMS,
        'num_features': len(feature_columns),
        'lag_steps': LAG_STEPS,
        'rolling_windows': ROLLING_WINDOWS,
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
        'num_features': len(feature_columns)
    })
    
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    print(f"Train RMSE: {metrics['train']['rmse']:.4f}, MAE: {metrics['train']['mae']:.4f}")
    print(f"Val RMSE:   {metrics['val']['rmse']:.4f}, MAE: {metrics['val']['mae']:.4f}")
    print(f"Test RMSE:  {metrics['test']['rmse']:.4f}, MAE: {metrics['test']['mae']:.4f}")


def main():
    """Main training pipeline."""
    # Initialize wandb
    run_name = f"catboost_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project="timeseries-forecast",
        name=run_name,
        config={
            'dataset': DATASET,
            'model': 'CatBoost',
            'catboost_params': CATBOOST_PARAMS,
            'num_iterations': NUM_ITERATIONS,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'lag_steps': LAG_STEPS,
            'rolling_windows': ROLLING_WINDOWS
        },
        tags=['baseline', 'catboost', DATASET]
    )
    
    print("="*60)
    print("CatBoost Baseline Training")
    print(f"Dataset: {DATASET}")
    print(f"wandb Project: timeseries-forecast")
    print(f"wandb Run: {run_name}")
    print("="*60)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare features
    train_df, val_df, test_df = prepare_features(train_df, val_df, test_df)
    
    # Prepare model data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, feature_columns = prepare_model_data(
        train_df, val_df, test_df
    )
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on all datasets
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    metrics = {}
    metrics['train'] = evaluate_model(model, X_train, y_train, scaler_y, 'Train')
    metrics['val'] = evaluate_model(model, X_val, y_val, scaler_y, 'Validation')
    metrics['test'] = evaluate_model(model, X_test, y_test, scaler_y, 'Test')
    
    # Save model and results
    save_model_and_results(model, metrics, feature_columns)
    
    # Finish wandb run
    wandb.finish()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()