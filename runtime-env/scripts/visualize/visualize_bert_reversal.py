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

# 데이터 로드 및 전처리
df = pd.read_csv(csv_file)
df['latency_mean_ms'] = pd.to_numeric(df['latency_mean_ms'], errors='coerce')

# BERT 모델 데이터만 추출
bert_df = df[df['model'] == 'bert'].copy()

# 시각화를 위한 특정 케이스 선택
# 1. IREE (CPU, O2)
# 2. IREE (CUDA, O2) - 역전 현상 발생 지점
# 3. PyTorch (CUDA) - 성능 베이스라인
plot_data = bert_df[
    ((bert_df['framework'] == 'IREE') & (bert_df['opt_level'] == 'O2')) |
    ((bert_df['framework'] == 'pytorch') & (bert_df['target'] == 'cuda'))
].copy()

# 레이블 생성을 위한 컬럼 추가
plot_data['display_label'] = plot_data.apply(
    lambda x: f"{x['framework'].upper()} ({x['target'].upper()})" + (f" - {x['opt_level']}" if x['framework'] == 'IREE' else ""), 
    axis=1
)

if plot_data.empty:
    print("⚠️ No relevant BERT data found for visualization.")
    exit()

# 스타일 설정
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 7))

# 시각화: BERT 성능 역전 현상 (IREE CPU vs IREE CUDA vs PyTorch CUDA)
colors = ['#3498db', '#e74c3c', '#2ecc71'] # Blue for CPU, Red for Problematic CUDA, Green for Baseline
plot = sns.barplot(data=plot_data, x='display_label', y='latency_mean_ms', palette=colors)

plt.title('BERT Performance Reversal: IREE CPU vs CUDA\n(Compared to PyTorch Baseline)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Latency (ms) - Lower is Better', fontsize=12)
plt.xlabel('Framework & Target Configuration', fontsize=12)

# 바 위에 수치 표시
for p in plot.patches:
    height = p.get_height()
    plot.annotate(f'{height:.2f} ms', 
                 (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='center', 
                 xytext=(0, 10), 
                 textcoords='offset points',
                 fontweight='bold')

plt.tight_layout()
output_path = os.path.join(output_dir, "bert_iree_reversal.png")
plt.savefig(output_path, dpi=300)
print(f"✅ BERT performance reversal graph saved to {output_path}")
