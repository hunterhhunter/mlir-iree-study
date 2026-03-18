import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create image directory if not exists
os.makedirs('img', exist_ok=True)

# 1. Load Data
df = pd.read_csv('benchmark_results.csv')

# 2. Filter for ONNX framework and normalize target names
onnx_df = df[df['framework'].str.lower() == 'onnx'].copy()
onnx_df['target'] = onnx_df['target'].replace({'cuda': 'GPU', 'gpu': 'GPU', 'cpu': 'CPU'})

# 3. Identify models where GPU latency is higher than CPU latency
pivot_df = onnx_df.pivot_table(index='model', columns='target', values='latency_mean_ms').reset_index()

# Filter models that have both CPU and GPU data, and where GPU > CPU
if 'GPU' in pivot_df.columns and 'CPU' in pivot_df.columns:
    reversal_models = pivot_df[
        (pivot_df['GPU'].notnull()) & 
        (pivot_df['CPU'].notnull()) & 
        (pivot_df['GPU'] > pivot_df['CPU'])
    ]['model'].tolist()
else:
    reversal_models = []

if not reversal_models:
    print("No ONNX GPU reversal cases found in the current dataset.")
else:
    plot_data = onnx_df[onnx_df['model'].isin(reversal_models)]

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Define color palette
    palette = {'CPU': '#3498db', 'GPU': '#e74c3c'}

    ax = sns.barplot(
        data=plot_data,
        x='model',
        y='latency_mean_ms',
        hue='target',
        palette=palette
    )

    # Add styling and labels
    plt.title('ONNX Runtime: GPU Latency Reversal Cases (Higher is Slower)', fontsize=15, pad=20)
    plt.ylabel('Mean Latency (ms)', fontsize=12)
    plt.xlabel('Model Type', fontsize=12)
    plt.legend(title='Device')

    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}ms', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = 'img/onnx_gpu_reversal.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    print(f"Detected Reversal Models: {reversal_models}")
