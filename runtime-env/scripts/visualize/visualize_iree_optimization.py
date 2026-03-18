import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 환경 설정
csv_file = "benchmark_results.csv"
output_dir = "img"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    exit()

df = pd.read_csv(csv_file)
df['latency_mean_ms'] = pd.to_numeric(df['latency_mean_ms'], errors='coerce')

# IREE 데이터만 추출
iree_df = df[df['framework'] == 'IREE'].copy()

if iree_df.empty:
    print("⚠️ No IREE data found for optimization level comparison.")
    exit()

# 스타일 설정
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

# 시각화: IREE O0 vs O1 vs O2
targets = ['cpu', 'cuda']
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

for i, target in enumerate(targets):
    target_iree = iree_df[iree_df['target'] == target].copy()
    if target_iree.empty: continue
    
    # O0, O1, O2 순서 보장
    target_iree['opt_level'] = pd.Categorical(target_iree['opt_level'], categories=['O0', 'O1', 'O2', 'O3'], ordered=True)
    
    plot = sns.barplot(data=target_iree, x='model', y='latency_mean_ms', hue='opt_level', ax=axes[i], palette='magma')
    axes[i].set_title(f'IREE Optimization Level Comparison on {target.upper()} (Lower is Better)', fontsize=16, fontweight='bold')
    axes[i].set_ylabel('Latency (ms)')
    axes[i].legend(title='Opt Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in plot.patches:
        if p.get_height() > 0:
            plot.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "iree_optimization_gap.png"), dpi=300)
print(f"✅ IREE optimization level graph saved to {output_dir}/viz_iree_optimization.png")
