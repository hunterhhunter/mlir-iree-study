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

# 프레임워크 라벨링 (IREE는 가장 성능이 좋은 O2만 대표로 사용하거나, 라벨에 표시)
def format_framework(row):
    if row['framework'] == 'IREE':
        return f"IREE ({row['opt_level']})"
    return row['framework']
df['Framework_Label'] = df.apply(format_framework, axis=1)

# 스타일 설정
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

# 시각화: CPU vs CUDA
targets = ['cpu', 'cuda']
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

for i, target in enumerate(targets):
    target_df = df[df['target'] == target].copy()
    if target_df.empty: continue
    
    plot = sns.barplot(data=target_df, x='model', y='latency_mean_ms', hue='Framework_Label', ax=axes[i], palette='viridis')
    axes[i].set_title(f'Mean Latency Comparison on {target.upper()} (Lower is Better)', fontsize=16, fontweight='bold')
    axes[i].set_ylabel('Latency (ms)')
    axes[i].legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in plot.patches:
        if p.get_height() > 0:
            plot.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "viz_framework_comparison.png"), dpi=300)
print(f"✅ Framework comparison graph saved to {output_dir}/viz_framework_comparison.png")
