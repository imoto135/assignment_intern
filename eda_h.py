"""
ETTh1 & ETTh2 データセットの高品質EDAスクリプト
企業報告用に、デザインに注力した可視化を生成します。

生成される分析:
1. 長期トレンドと季節性の可視化
2. 日内変動パターンの比較
3. 相関分析（ヒートマップ）
4. 統計サマリー
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# デザイン設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.edgecolor'] = '#dee2e6'
plt.rcParams['grid.color'] = '#dee2e6'
plt.rcParams['grid.alpha'] = 0.5

# パラメータ
DATA_FILES = {
    'ETTh1': 'data/ETTh1.csv',
    'ETTh2': 'data/ETTh2.csv'
}
OUTPUT_DIR = Path('eda_output_h')
COLORS = {
    'ETTh1': '#2E86AB',  # 深い青
    'ETTh2': '#A23B72'   # 深い紫
}


def load_data(filepath):
    """データの読み込みと前処理"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def plot_long_term_trend(datasets, output_dir):
    """
    長期トレンドと季節性の可視化
    2年間の推移を滑らかに表示
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # 元データ（薄く）
        ax.plot(df.index, df['OT'], alpha=0.3, linewidth=0.5, 
                color=COLORS[name], label='実測値')
        
        # 7日移動平均（トレンド）
        rolling_7d = df['OT'].rolling(window=24*7, center=True).mean()
        ax.plot(df.index, rolling_7d, linewidth=2.5, 
                color=COLORS[name], label='7日移動平均（トレンド）')
        
        # 月次平均（季節性）
        monthly_avg = df['OT'].resample('M').mean()
        ax.scatter(monthly_avg.index, monthly_avg.values, 
                  s=100, color=COLORS[name], alpha=0.7, 
                  edgecolors='white', linewidth=2, zorder=5,
                  label='月次平均')
        
        # デザイン調整
        ax.set_title(f'{name} - オイル温度の長期推移（2016-2018）', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('日時', fontsize=12)
        ax.set_ylabel('オイル温度 (℃)', fontsize=12)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 統計情報の追加
        mean_val = df['OT'].mean()
        std_val = df['OT'].std()
        ax.axhline(mean_val, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.6, label=f'平均: {mean_val:.2f}℃')
        ax.fill_between(df.index, mean_val - std_val, mean_val + std_val,
                        alpha=0.1, color='red', label=f'±1σ範囲')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trend_seasonal.png', dpi=300, bbox_inches='tight')
    print(f"✓ 長期トレンド図を保存: trend_seasonal.png")
    plt.close()


def plot_daily_patterns(datasets, output_dir):
    """
    日内変動パターンの比較
    時間帯別の平均値と信頼区間を表示
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # 時間別の統計量
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_stats = df_copy.groupby('hour')['OT'].agg(['mean', 'std', 'min', 'max'])
        
        hours = hourly_stats.index
        means = hourly_stats['mean']
        stds = hourly_stats['std']
        
        # メインライン
        ax.plot(hours, means, linewidth=3, color=COLORS[name], 
               marker='o', markersize=8, label='平均値')
        
        # 信頼区間（±1σ）
        ax.fill_between(hours, means - stds, means + stds, 
                        alpha=0.2, color=COLORS[name], label='±1σ範囲')
        
        # 最小・最大範囲
        ax.fill_between(hours, hourly_stats['min'], hourly_stats['max'],
                        alpha=0.05, color=COLORS[name], label='最小〜最大範囲')
        
        # ピーク時刻のハイライト
        peak_hour = means.idxmax()
        peak_temp = means.max()
        ax.scatter([peak_hour], [peak_temp], s=300, color='red', 
                  edgecolors='white', linewidth=3, zorder=10, marker='*')
        
        
        # デザイン調整
        ax.set_title(f'{name} - 時間帯別オイル温度パターン', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('時刻', fontsize=12)
        ax.set_ylabel('オイル温度 (℃)', fontsize=12)
        ax.set_xticks(range(0, 24, 2))
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'daily_patterns.png', dpi=300, bbox_inches='tight')
    print(f"✓ 日内変動図を保存: daily_patterns.png")
    plt.close()


def plot_correlation_comparison(datasets, output_dir):
    """
    相関分析の比較（ヒートマップ）
    h1とh2の相関構造の違いを可視化
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # 相関行列
        corr = df.corr()
        
        # マスク（上三角を隠す）
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # ヒートマップ
        sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdBu_r', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax,
                   annot_kws={'size': 10, 'weight': 'bold'})
        
        # OT行をハイライト
        ot_idx = list(corr.columns).index('OT')
        ax.add_patch(plt.Rectangle((0, ot_idx), len(corr.columns), 1, 
                                   fill=False, edgecolor='gold', lw=3))
        
        ax.set_title(f'{name} - 変数間相関マトリクス', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ 相関ヒートマップを保存: correlation_heatmap.png")
    plt.close()


def plot_ot_correlation_bar(datasets, output_dir):
    """
    OTとの相関係数を棒グラフで比較
    h1とh2の違いを明確に
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # データ準備
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    x = np.arange(len(features))
    width = 0.35
    
    corr_h1 = datasets['ETTh1'].corr()['OT'].drop('OT')[features]
    corr_h2 = datasets['ETTh2'].corr()['OT'].drop('OT')[features]
    
    # 棒グラフ
    bars1 = ax.bar(x - width/2, corr_h1, width, label='ETTh1', 
                   color=COLORS['ETTh1'], alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, corr_h2, width, label='ETTh2', 
                   color=COLORS['ETTh2'], alpha=0.8, edgecolor='white', linewidth=2)
    
    # 値ラベル
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    # デザイン調整
    ax.set_title('オイル温度（OT）との相関係数比較', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('特徴量', fontsize=12)
    ax.set_ylabel('相関係数', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=11)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ot_correlation_bar.png', dpi=300, bbox_inches='tight')
    print(f"✓ OT相関棒グラフを保存: ot_correlation_bar.png")
    plt.close()


def generate_summary_report(datasets, output_dir):
    """
    統計サマリーレポートの生成
    """
    with open(output_dir / 'summary_report.md', 'w', encoding='utf-8') as f:
        f.write("# ETTh1 & ETTh2 データセット分析サマリー\n\n")
        f.write("## データ概要\n\n")
        
        for name, df in datasets.items():
            f.write(f"### {name}\n\n")
            f.write(f"- **期間**: {df.index.min()} 〜 {df.index.max()}\n")
            f.write(f"- **データ数**: {len(df):,} 行\n")
            f.write(f"- **サンプリング間隔**: 1時間\n")
            f.write(f"- **欠損値**: {df.isnull().sum().sum()} 件\n\n")
            
            f.write("#### オイル温度（OT）統計量\n\n")
            stats = df['OT'].describe()
            f.write(f"- 平均: {stats['mean']:.2f}℃\n")
            f.write(f"- 標準偏差: {stats['std']:.2f}℃\n")
            f.write(f"- 最小値: {stats['min']:.2f}℃\n")
            f.write(f"- 最大値: {stats['max']:.2f}℃\n")
            f.write(f"- 範囲: {stats['max'] - stats['min']:.2f}℃\n\n")
            
            f.write("#### OTとの相関係数（上位3変数）\n\n")
            corr_ot = df.corr()['OT'].drop('OT').sort_values(ascending=False)
            for i, (var, val) in enumerate(corr_ot.head(3).items(), 1):
                f.write(f"{i}. **{var}**: {val:.4f}\n")
            f.write("\n")
    
    print(f"✓ サマリーレポートを保存: summary_report.md")


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("ETTh1 & ETTh2 高品質EDA実行中...")
    print("="*80 + "\n")
    
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    datasets = {}
    for name, filepath in DATA_FILES.items():
        print(f"[{name}] データ読み込み中...")
        datasets[name] = load_data(filepath)
    
    print("\n可視化生成中...\n")
    
    # 各種プロット生成
    plot_long_term_trend(datasets, OUTPUT_DIR)
    plot_daily_patterns(datasets, OUTPUT_DIR)
    plot_correlation_comparison(datasets, OUTPUT_DIR)
    plot_ot_correlation_bar(datasets, OUTPUT_DIR)
    
    # サマリーレポート
    generate_summary_report(datasets, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(f"✅ すべての分析が完了しました。結果は '{OUTPUT_DIR}' に保存されています。")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
