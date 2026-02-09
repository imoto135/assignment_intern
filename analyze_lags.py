
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Parameters
DATA_PATH = 'data/ETTh1.csv'
OUTPUT_DIR = Path('eda_output')
LAGS = [1, 2, 3, 6, 12, 24]  # Hours

def analyze_lags():
    # Load Data
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Target variable
    target = 'OT'
    
    # Generate Lag Features
    print(f"Generating lag features for {LAGS} hours...")
    
    lagged_df = pd.DataFrame(index=df.index)
    lagged_df['Target_OT'] = df[target]
    
    features = [col for col in df.columns]
    
    for feature in features:
        for lag in LAGS:
            lagged_df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
            
    # Drop NaN values created by shifting
    lagged_df = lagged_df.dropna()
    
    # Calculate Correlations
    print("Calculating correlations...")
    correlations = lagged_df.corr()['Target_OT'].drop('Target_OT')
    
    # Sort by absolute correlation
    sorted_corrs = correlations.abs().sort_values(ascending=False)
    
    # Display Top 15 Correlated Features
    print("\nTop 15 Most Correlated Features:")
    print(sorted_corrs.head(15))
    
    # Visualization
    plt.figure(figsize=(10, 8))
    top_features = sorted_corrs.head(20).index
    sns.barplot(x=correlations[top_features], y=top_features, palette='viridis')
    plt.title(f'Top 20 Lag Feature Correlations with Target (OT)', fontsize=14)
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lag_correlations.png')
    print(f"\nCorrelation plot saved to {OUTPUT_DIR / 'lag_correlations.png'}")

if __name__ == "__main__":
    analyze_lags()
