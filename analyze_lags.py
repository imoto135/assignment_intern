
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.stattools import ccf

# Parameters
DATA_PATH = 'data/ETTh1.csv'
OUTPUT_DIR = Path('eda_lags')
MAX_LAG = 72  # Hours (3 days)

def analyze_lags_detailed():
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    target = 'OT'
    features = [col for col in df.columns if col != target]
    
    print(f"Analyzing lags (Max: {MAX_LAG} hours)...")

    # 1. Autocorrelation (ACF) of Target
    # ------------------------------------------------
    plt.figure(figsize=(12, 5))
    pd.plotting.autocorrelation_plot(df[target])
    plt.title(f'Autocorrelation of {target} (Self-Correlation)')
    plt.xlim(0, MAX_LAG * 2)  # Show up to double the max lag
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'autocorrelation_OT.png')
    print("✓ Saved autocorrelation plot")

    # 2. Cross-Correlation between Features and Target
    # ------------------------------------------------
    # Shows how much a feature at t-k correlates with Target at t
    
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 3*len(features)), sharex=True)
    if len(features) == 1: axes = [axes]
    
    lags = np.arange(MAX_LAG + 1)
    
    for idx, feat in enumerate(features):
        # Calculate cross-correlation
        # ccf returns correlation at lag 0, 1, 2...
        cross_corr = ccf(df[target], df[feat], adjusted=False)[:MAX_LAG+1]
        
        # Determine max correlation lag
        max_corr_idx = np.argmax(np.abs(cross_corr))
        max_corr_val = cross_corr[max_corr_idx]
        
        axes[idx].bar(lags, cross_corr, width=0.5, alpha=0.7)
        axes[idx].axhline(0, color='black', linewidth=0.5)
        
        # Highlight max correlation
        axes[idx].plot(max_corr_idx, max_corr_val, 'ro')
        axes[idx].text(max_corr_idx, max_corr_val + 0.05, 
                      f'Lag {max_corr_idx}h\n({max_corr_val:.2f})', 
                      ha='center', fontsize=9, color='red')
        
        axes[idx].set_title(f'Cross-Correlation: {target}(t) vs {feat}(t-k)', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Correlation')
        axes[idx].grid(True, alpha=0.3)
        
        # Add 24h cycle markers
        for h in [24, 48, 72]:
            if h <= MAX_LAG:
                axes[idx].axvline(h, color='green', linestyle='--', alpha=0.5)

    plt.xlabel('Lag (Hours)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cross_correlation_features.png')
    print("✓ Saved cross-correlation plot")

    # 3. Lag Correlation Heatmap (Selected Lags)
    # ------------------------------------------------
    # Create lag features for heatmap
    selected_lags = [1, 3, 6, 12, 24, 48]
    lagged_df = pd.DataFrame()
    lagged_df['Target_OT'] = df[target]
    
    # Add lags for target and top features (e.g., MULL, HULL)
    for lag in selected_lags:
        lagged_df[f'OT_lag_{lag}'] = df[target].shift(lag)
        if 'MULL' in df.columns:
            lagged_df[f'MULL_lag_{lag}'] = df['MULL'].shift(lag)
        if 'HULL' in df.columns:
            lagged_df[f'HULL_lag_{lag}'] = df['HULL'].shift(lag)
            
    # Compute correlation matrix
    corr_matrix = lagged_df.corr()
    
    # Plot Heatmap
    plt.figure(figsize=(14, 10))
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                mask=mask)
    
    plt.title('Correlation Matrix of Lagged Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lag_heatmap.png')
    print("✓ Saved lag heatmap")

    # 4. Save Summary Report
    # ------------------------------------------------
    with open(OUTPUT_DIR / 'lag_analysis_summary.txt', 'w') as f:
        f.write("# Lag Analysis Summary\n\n")
        f.write("## Max Correlation Lags\n")
        f.write("| Feature | Optimal Lag (Hours) | Correlation |\n")
        f.write("|---|---|---|\n")
        
        for feat in features:
            cc = ccf(df[target], df[feat], adjusted=False)[:MAX_LAG+1]
            idx = np.argmax(np.abs(cc))
            val = cc[idx]
            f.write(f"| {feat} | {idx} | {val:.4f} |\n")
            
    print(f"\nAnalysis complete. Results saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    analyze_lags_detailed()
