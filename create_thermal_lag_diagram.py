
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 図の作成
fig = plt.figure(figsize=(16, 10))

# ===== 上部: タイムライン =====
ax1 = plt.subplot(3, 1, 1)
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 2)
ax1.axis('off')

# タイムライン
ax1.plot([0, 24], [1, 1], 'k-', linewidth=3)
for h in range(0, 25, 2):
    ax1.plot([h, h], [0.9, 1.1], 'k-', linewidth=2)
    ax1.text(h, 0.7, f'{h}時', ha='center', fontsize=10)

# 負荷増加（10時）
ax1.annotate('', xy=(10, 1.5), xytext=(10, 1.1),
            arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
ax1.text(10, 1.7, '負荷増加\n(MULL↑)', ha='center', fontsize=12, 
        fontweight='bold', color='blue',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='blue', linewidth=2))

# 温度ピーク（20時）
ax1.annotate('', xy=(20, 1.5), xytext=(20, 1.1),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax1.text(20, 1.7, '温度ピーク\n(OT↑)', ha='center', fontsize=12, 
        fontweight='bold', color='red',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='red', linewidth=2))

# 10時間の遅れを示す矢印
arrow = FancyArrowPatch((10, 0.5), (20, 0.5),
                       arrowstyle='<->', mutation_scale=30, 
                       linewidth=3, color='purple')
ax1.add_patch(arrow)
ax1.text(15, 0.3, '約10時間の遅れ（熱遅れ効果）', ha='center', 
        fontsize=13, fontweight='bold', color='purple')

ax1.set_title('熱遅れ効果のメカニズム', fontsize=16, fontweight='bold', pad=20)

# ===== 中部: 変圧器の構造と熱の流れ =====
ax2 = plt.subplot(3, 1, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.axis('off')

# 変圧器の簡易図
# コイル
coil = FancyBboxPatch((2, 2), 1.5, 2, boxstyle="round,pad=0.1", 
                      facecolor='orange', edgecolor='darkorange', linewidth=3)
ax2.add_patch(coil)
ax2.text(2.75, 3, 'コイル\n（発熱源）', ha='center', va='center', 
        fontsize=11, fontweight='bold')

# 鉄心
core = FancyBboxPatch((4, 2), 1.5, 2, boxstyle="round,pad=0.1",
                     facecolor='gray', edgecolor='black', linewidth=3)
ax2.add_patch(core)
ax2.text(4.75, 3, '鉄心', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='white')

# オイル
oil = FancyBboxPatch((6.5, 1.5), 2, 3, boxstyle="round,pad=0.1",
                    facecolor='lightblue', edgecolor='blue', linewidth=3)
ax2.add_patch(oil)
ax2.text(7.5, 3, 'オイル\n（測定点）', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='darkblue')

# 熱の流れを示す矢印
arrow1 = FancyArrowPatch((3.5, 3), (4, 3),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color='red')
ax2.add_patch(arrow1)
ax2.text(3.75, 3.5, '数時間', ha='center', fontsize=9, color='red')

arrow2 = FancyArrowPatch((5.5, 3), (6.5, 3),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color='red')
ax2.add_patch(arrow2)
ax2.text(6, 3.5, '数時間', ha='center', fontsize=9, color='red')

ax2.text(5, 4.5, '熱の伝達経路（合計: 約10時間）', ha='center', 
        fontsize=13, fontweight='bold')

# ===== 下部: グラフ =====
ax3 = plt.subplot(3, 1, 3)

# 時間軸
hours = np.arange(0, 24, 0.5)

# 負荷のパターン（10時にピーク）
load = np.zeros_like(hours)
for i, h in enumerate(hours):
    if 8 <= h <= 12:
        load[i] = 100 * np.exp(-((h - 10)**2) / 2)
    else:
        load[i] = 20

# 温度のパターン（20時にピーク、10時間遅れ）
temp = np.zeros_like(hours)
for i, h in enumerate(hours):
    if 18 <= h <= 22:
        temp[i] = 30 + 10 * np.exp(-((h - 20)**2) / 2)
    else:
        temp[i] = 25

# プロット
ax3.plot(hours, load, 'b-', linewidth=3, label='負荷（MULL）', marker='o', markersize=4)
ax3_twin = ax3.twinx()
ax3_twin.plot(hours, temp, 'r-', linewidth=3, label='オイル温度（OT）', marker='s', markersize=4)

# 10時と20時に縦線
ax3.axvline(10, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(20, color='red', linestyle='--', linewidth=2, alpha=0.7)

# ラベル
ax3.set_xlabel('時刻', fontsize=12, fontweight='bold')
ax3.set_ylabel('負荷（任意単位）', fontsize=12, fontweight='bold', color='blue')
ax3_twin.set_ylabel('オイル温度（℃）', fontsize=12, fontweight='bold', color='red')
ax3.set_title('負荷と温度の時間変化（10時間の遅れ）', fontsize=14, fontweight='bold', pad=15)

# グリッド
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 24)
ax3.set_xticks(range(0, 25, 2))

# 凡例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

# 注釈
ax3.text(10, 90, '負荷ピーク', ha='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax3_twin.text(20, 39, '温度ピーク', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.tight_layout()
plt.savefig('eda_lags/thermal_lag_explanation.png', dpi=300, bbox_inches='tight')
print("✓ 熱遅れ効果の説明図を保存: eda_lags/thermal_lag_explanation.png")
plt.close()
