"""
ETT (Electricity Transformer Temperature) データセットの基本的なEDAスクリプト

このスクリプトは以下の分析を実行します:
1. データの基本情報と統計量
2. 欠損値の確認
3. 時系列データの可視化
4. 相関分析
5. 分布の確認
6. 季節性・トレンドの分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（macOS用）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 出力ディレクトリの作成
OUTPUT_DIR = Path('eda_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# データファイルのパス
DATA_FILES = {
    'ETTh1': 'data/ETTh1.csv',
    'ETTh2': 'data/ETTh2.csv',
    'ETTm1': 'data/ETTm1.csv',
    'ETTm2': 'data/ETTm2.csv'
}


def load_data(filepath):
    """データの読み込みと前処理"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def basic_info(df, dataset_name):
    """基本情報の表示"""
    print(f"\n{'='*60}")
    print(f"データセット: {dataset_name}")
    print(f"{'='*60}")
    print(f"\n【データ形状】")
    print(f"行数: {df.shape[0]:,}")
    print(f"列数: {df.shape[1]}")
    print(f"期間: {df.index.min()} ～ {df.index.max()}")
    print(f"データ間隔: {df.index.to_series().diff().mode()[0]}")
    
    print(f"\n【カラム情報】")
    print(df.dtypes)
    
    print(f"\n【欠損値】")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("欠損値なし")
    else:
        print(missing[missing > 0])
    
    print(f"\n【基本統計量】")
    print(df.describe())
    
    return df.describe()


def plot_time_series(df, dataset_name):
    """時系列データの可視化"""
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(15, 3*len(df.columns)))
    
    if len(df.columns) == 1:
        axes = [axes]
    
    for idx, col in enumerate(df.columns):
        axes[idx].plot(df.index, df[col], linewidth=0.5, alpha=0.8)
        axes[idx].set_title(f'{col} の時系列推移', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('日時')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 時系列グラフを保存: {dataset_name}_timeseries.png")


def plot_correlation(df, dataset_name):
    """相関分析"""
    # 相関行列の計算
    corr_matrix = df.corr()
    
    # ヒートマップの作成
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'{dataset_name} - 相関行列', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 相関行列を保存: {dataset_name}_correlation.png")
    
    # OTとの相関（OTが存在する場合）
    if 'OT' in df.columns:
        print(f"\n【OT（オイル温度）との相関】")
        ot_corr = corr_matrix['OT'].sort_values(ascending=False)
        print(ot_corr)


def plot_distributions(df, dataset_name):
    """分布の確認"""
    n_cols = len(df.columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(df.columns):
        axes[idx].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col} の分布', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('頻度')
        axes[idx].grid(True, alpha=0.3)
        
        # 統計情報を追加
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中央値: {median_val:.2f}')
        axes[idx].legend(fontsize=8)
    
    # 余分なサブプロットを非表示
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 分布グラフを保存: {dataset_name}_distributions.png")


def plot_boxplots(df, dataset_name):
    """箱ひげ図による外れ値の確認"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # データの正規化（スケールが異なるため）
    df_normalized = (df - df.mean()) / df.std()
    
    df_normalized.boxplot(ax=ax)
    ax.set_title(f'{dataset_name} - 箱ひげ図（標準化後）', fontsize=14, fontweight='bold')
    ax.set_xlabel('変数')
    ax.set_ylabel('標準化された値')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 箱ひげ図を保存: {dataset_name}_boxplots.png")


def seasonal_decomposition_simple(df, dataset_name, target_col='OT'):
    """簡易的な季節性分析（移動平均を使用）"""
    if target_col not in df.columns:
        print(f"警告: {target_col} が見つかりません。季節性分析をスキップします。")
        return
    
    # 移動平均でトレンドを抽出
    window_size = 24 if 'h' in dataset_name.lower() else 24*7  # 時間単位なら24時間、分単位なら1週間
    
    data = df[target_col].copy()
    trend = data.rolling(window=window_size, center=True).mean()
    detrended = data - trend
    seasonal = detrended.rolling(window=window_size, center=True).mean()
    residual = data - trend - seasonal
    
    # プロット
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    axes[0].plot(data.index, data, linewidth=0.5)
    axes[0].set_title(f'{target_col} - 元データ', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(target_col)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(trend.index, trend, linewidth=1, color='orange')
    axes[1].set_title('トレンド成分', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('トレンド')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(seasonal.index, seasonal, linewidth=0.5, color='green')
    axes[2].set_title('季節性成分', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('季節性')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(residual.index, residual, linewidth=0.5, color='red', alpha=0.5)
    axes[3].set_title('残差成分', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('残差')
    axes[3].set_xlabel('日時')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 季節性分解グラフを保存: {dataset_name}_seasonal_decomposition.png")


def analyze_daily_patterns(df, dataset_name, target_col='OT'):
    """日次パターンの分析（時間単位データの場合）"""
    if target_col not in df.columns:
        return
    
    # 時間単位のデータかチェック
    if 'h' not in dataset_name.lower():
        print(f"情報: {dataset_name} は時間単位ではないため、日次パターン分析をスキップします。")
        return
    
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 時間別の平均
    hourly_avg = df_copy.groupby('hour')[target_col].mean()
    axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[0].set_title(f'{target_col} - 時間別平均', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('時刻')
    axes[0].set_ylabel(f'平均{target_col}')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2))
    
    # 曜日別の平均
    dow_avg = df_copy.groupby('day_of_week')[target_col].mean()
    dow_labels = ['月', '火', '水', '木', '金', '土', '日']
    axes[1].bar(range(7), dow_avg.values, alpha=0.7, edgecolor='black')
    axes[1].set_title(f'{target_col} - 曜日別平均', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('曜日')
    axes[1].set_ylabel(f'平均{target_col}')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_labels)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 月別の平均
    monthly_avg = df_copy.groupby('month')[target_col].mean()
    axes[2].plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2, color='green')
    axes[2].set_title(f'{target_col} - 月別平均', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('月')
    axes[2].set_ylabel(f'平均{target_col}')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{dataset_name}_daily_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 日次パターングラフを保存: {dataset_name}_daily_patterns.png")


def generate_summary_report(all_stats, output_file='eda_summary.txt'):
    """サマリーレポートの生成"""
    with open(OUTPUT_DIR / output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ETTデータセット EDAサマリーレポート\n")
        f.write("="*80 + "\n\n")
        
        for dataset_name, stats in all_stats.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"データセット: {dataset_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(stats.to_string())
            f.write("\n\n")
    
    print(f"\n✓ サマリーレポートを保存: {output_file}")


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("ETT (Electricity Transformer Temperature) データセット EDA")
    print("="*80)
    
    all_stats = {}
    
    # 各データセットに対してEDAを実行
    for dataset_name, filepath in DATA_FILES.items():
        try:
            # データ読み込み
            df = load_data(filepath)
            
            # 基本情報
            stats = basic_info(df, dataset_name)
            all_stats[dataset_name] = stats
            
            # 可視化
            print(f"\n【可視化処理中...】")
            plot_time_series(df, dataset_name)
            plot_correlation(df, dataset_name)
            plot_distributions(df, dataset_name)
            plot_boxplots(df, dataset_name)
            seasonal_decomposition_simple(df, dataset_name)
            analyze_daily_patterns(df, dataset_name)
            
            print(f"\n✓ {dataset_name} の分析完了\n")
            
        except FileNotFoundError:
            print(f"警告: {filepath} が見つかりません。スキップします。")
        except Exception as e:
            print(f"エラー: {dataset_name} の処理中にエラーが発生しました: {e}")
    
    # サマリーレポート生成
    if all_stats:
        generate_summary_report(all_stats)
    
    print("\n" + "="*80)
    print(f"すべての分析が完了しました。結果は '{OUTPUT_DIR}' ディレクトリに保存されています。")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
