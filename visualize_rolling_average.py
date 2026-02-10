
import pandas as pd
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
df = pd.read_csv('data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 移動平均の計算
df['MULL_rolling_mean_12'] = df['MULL'].rolling(window=12).mean()
df['OT_rolling_mean_12'] = df['OT'].rolling(window=12).mean()

# 1週間分のデータを抽出（見やすくするため）
sample = df['2016-07-01':'2016-07-07']

# プロット
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 上段: MULL（負荷）
ax1 = axes[0]
ax1.plot(sample.index, sample['MULL'], 'b-', alpha=0.5, linewidth=1, label='MULL（元データ）')
ax1.plot(sample.index, sample['MULL_rolling_mean_12'], 'r-', linewidth=3, label='MULL 12時間移動平均（蓄積負荷）')
ax1.set_title('中圧無効負荷（MULL）: 元データ vs 移動平均', fontsize=14, fontweight='bold')
ax1.set_ylabel('負荷', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 下段: OT（温度）
ax2 = axes[1]
ax2.plot(sample.index, sample['OT'], 'g-', alpha=0.5, linewidth=1, label='OT（元データ）')
ax2.plot(sample.index, sample['OT_rolling_mean_12'], 'orange', linewidth=3, label='OT 12時間移動平均')
ax2.set_title('オイル温度（OT）: 元データ vs 移動平均', fontsize=14, fontweight='bold')
ax2.set_xlabel('日時', fontsize=12)
ax2.set_ylabel('温度（℃）', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_lags/rolling_average_explanation.png', dpi=300, bbox_inches='tight')
print("✓ 移動平均の説明図を保存: eda_lags/rolling_average_explanation.png")
plt.close()

# 相関の確認
print("\n【移動平均と温度の相関】")
print("="*50)
print(f"MULL（元データ）と OT の相関: {df['MULL'].corr(df['OT']):.4f}")
print(f"MULL_rolling_mean_12 と OT の相関: {df['MULL_rolling_mean_12'].corr(df['OT']):.4f}")
print("="*50)
