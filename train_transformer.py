"""
Transformer model training script for time series forecasting on ETTh1 dataset.
Comparable with LightGBM baseline.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
OUTPUT_DIR = Path('experiments/transformer')
DATASET = 'ETTh1'
TARGET_COL = 'OT'
FEATURE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# Feature engineering parameters (同じ特徴量を使用)
LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

# Transformer parameters
SEQ_LEN = 24  # 入力シーケンス長（24時間）
PRED_LEN = 1   # 出力長（1時間先を予測）
D_MODEL = 64   # Transformer の埋め込み次元
NUM_HEADS = 4  # マルチヘッドアテンション
NUM_LAYERS = 2 # Transformer レイヤー数
D_FF = 256     # フィードフォワードネットワークの次元
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


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


def create_sequences(data, target_col, seq_len=24):
    """Create sequences for Transformer."""
    X, y = [], []
    
    # 数値列のみを選択（日付などの非数値列を除外）
    numeric_data = data.select_dtypes(include=[np.number])
    data_values = numeric_data.values
    target_idx = list(numeric_data.columns).index(target_col) if target_col in numeric_data.columns else -1
    
    if target_idx == -1:
        raise ValueError(f"Target column '{target_col}' not found in numeric data")
    
    for i in range(len(data_values) - seq_len):
        X.append(data_values[i:i+seq_len])  # 全特徴量を使用
        y.append(data_values[i+seq_len, target_idx])  # ターゲット列のみ
    
    return np.array(X), np.array(y)


def prepare_datasets(train_df, val_df, test_df):
    """Prepare datasets for Transformer."""
    print("\nPreparing datasets for Transformer...")
    
    # Reset index to avoid including datetime index in values
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # 数値列のみを選択
    train_df = train_df.select_dtypes(include=[np.number])
    val_df = val_df.select_dtypes(include=[np.number])
    test_df = test_df.select_dtypes(include=[np.number])
    
    # Get feature columns
    feature_cols = list(train_df.columns)
    
    # シーケンスを作成
    X_train, y_train = create_sequences(train_df, TARGET_COL, SEQ_LEN)
    X_val, y_val = create_sequences(val_df, TARGET_COL, SEQ_LEN)
    X_test, y_test = create_sequences(test_df, TARGET_COL, SEQ_LEN)
    
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Convert to float to ensure all data is numeric
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaler
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
    
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"  Data normalized with StandardScaler")
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, scaler_y, feature_cols)


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_size, d_model, num_heads, num_layers, d_ff, dropout, pred_len=1):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, pred_len)
        self.d_model = d_model
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use last token
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(X_train, y_train, X_val, y_val):
    """Train Transformer model with logging."""
    print("\n" + "="*60)
    print("Training Transformer Model")
    print("="*60)
    
    # Create logs directory
    LOGS_DIR = Path('experiments/transformer/logs')
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = TransformerModel(
        input_size=X_train.shape[-1],
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pred_len=PRED_LEN
    ).to(DEVICE)
    
    print(f"\nModel Architecture:")
    print(f"  Input size: {X_train.shape[-1]}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  D_model: {D_MODEL}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # ログファイルの準備
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f'training_log_{timestamp}.csv'
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Training logs will be saved to: {log_file}")
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # ログに記録
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{current_lr:.6f}'])
        
        # Log to wandb
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'epoch': epoch
        }, step=epoch)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), LOGS_DIR / f'best_model_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(LOGS_DIR / f'best_model_{timestamp}.pt'))
    
    print(f"\nBest epoch: {best_epoch+1}, Best val loss: {best_val_loss:.6f}")
    
    return model, log_file


def predict(model, X, device):
    """Make predictions."""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        y_pred = model(X_tensor)
    
    return y_pred.cpu().numpy()


def evaluate_model(model, X, y, scaler_y, dataset_name='Test'):
    """Evaluate model performance."""
    print(f"\n{dataset_name} Evaluation:")
    print("-" * 40)
    
    y_pred = predict(model, X, DEVICE)
    
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


def save_model_and_results(model, metrics, feature_columns, log_file):
    """Save trained model and results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = OUTPUT_DIR / f'transformer_{DATASET}_{timestamp}.pt'
    torch.save(model.state_dict(), str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Copy log file to output directory
    import shutil
    output_log = OUTPUT_DIR / f'training_log_{DATASET}_{timestamp}.csv'
    shutil.copy(log_file, output_log)
    print(f"Training log saved to: {output_log}")
    
    # Save metrics and configuration
    results = {
        'dataset': DATASET,
        'timestamp': timestamp,
        'model_type': 'Transformer',
        'parameters': {
            'seq_len': SEQ_LEN,
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS
        },
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
    run_name = f"transformer_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project="timeseries-forecast",
        name=run_name,
        config={
            'dataset': DATASET,
            'model': 'Transformer',
            'seq_len': SEQ_LEN,
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'lag_steps': LAG_STEPS,
            'rolling_windows': ROLLING_WINDOWS
        },
        tags=['baseline', 'transformer', DATASET]
    )
    
    print("="*60)
    print("Transformer Baseline Training")
    print(f"Dataset: {DATASET}")
    print(f"wandb Project: timeseries-forecast")
    print(f"wandb Run: {run_name}")
    print("="*60)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare features
    train_df, val_df, test_df = prepare_features(train_df, val_df, test_df)
    
    # Prepare datasets
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, feature_columns = prepare_datasets(
        train_df, val_df, test_df
    )
    
    # Train model
    model, log_file = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on all datasets
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    metrics = {}
    metrics['train'] = evaluate_model(model, X_train, y_train, scaler_y, 'Train')
    metrics['val'] = evaluate_model(model, X_val, y_val, scaler_y, 'Validation')
    metrics['test'] = evaluate_model(model, X_test, y_test, scaler_y, 'Test')
    
    # Save model and results
    save_model_and_results(model, metrics, feature_columns, log_file)
    
    # Finish wandb run
    wandb.finish()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()