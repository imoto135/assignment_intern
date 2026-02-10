
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
print("データ読み込み中...")
df = pd.read_csv('data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# パラメータ
MAX_LAG = 48  # 48時間（2日分）
target = 'OT'

# 相関が高い主要変数
key_features = ['HULL', 'MULL', 'HUFL', 'MUFL']

# 図の作成
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    # 相互相関の計算
    cross_corr = ccf(df[target], df[feature], adjusted=False)[:MAX_LAG+1]
    lags = np.arange(MAX_LAG + 1)
    
    # 最大相関のラグを特定
    max_corr_idx = np.argmax(np.abs(cross_corr))
    max_corr_val = cross_corr[max_corr_idx]
    
    # 棒グラフ
    colors = ['lightblue' if i != max_corr_idx else 'red' for i in range(len(lags))]
    ax.bar(lags, cross_corr, width=0.8, color=colors, edgecolor='black', linewidth=0.5)
    
    # ゼロライン
    ax.axhline(0, color='black', linewidth=1)
    
    # 最大相関点を強調
    ax.plot(max_corr_idx, max_corr_val, 'r*', markersize=20, 
           markeredgecolor='darkred', markeredgewidth=2, zorder=10)
    
    # 注釈（すべて上側に配置）
    y_offset = 0.05
    
    ax.text(max_corr_idx, max_corr_val + y_offset, 
           f'{max_corr_idx}時間\n相関: {max_corr_val:.3f}',
           ha='center', fontsize=12, fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='red', linewidth=2, alpha=0.9))
    
    # 24時間サイクルのマーカー
    for h in [24, 48]:
        if h <= MAX_LAG:
            ax.axvline(h, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(h, ax.get_ylim()[1] * 0.9, f'{h}h', 
                   ha='center', fontsize=9, color='green')
    
    # タイトルと軸ラベル
    ax.set_title(f'{feature} と OT の相互相関', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('ラグ（時間）', fontsize=11)
    ax.set_ylabel('相関係数', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-1, MAX_LAG + 1)
    
    # 変数の説明を追加
    descriptions = {
        'HULL': '高圧側 無効負荷',
        'MULL': '中圧側 無効負荷',
        'HUFL': '高圧側 有効負荷',
        'MUFL': '中圧側 有効負荷'
    }
    ax.text(0.02, 0.98, descriptions[feature], 
           transform=ax.transAxes, fontsize=10, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('ETTh1: 各負荷変数とオイル温度（OT）の時間遅れ相関', 
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('eda_lags/cross_correlation_key_features.png', dpi=300, bbox_inches='tight')
print("✓ 相互相関グラフを保存: eda_lags/cross_correlation_key_features.png")

# サマリーをテキストで出力
print("\n【相互相関分析の結果】")
print("="*60)
for feature in key_features:
    cc = ccf(df[target], df[feature], adjusted=False)[:MAX_LAG+1]
    max_idx = np.argmax(np.abs(cc))
    max_val = cc[max_idx]
    print(f"{feature:6s} | 最大相関ラグ: {max_idx:2d}時間 | 相関係数: {max_val:+.4f}")
print("="*60)

plt.close()
