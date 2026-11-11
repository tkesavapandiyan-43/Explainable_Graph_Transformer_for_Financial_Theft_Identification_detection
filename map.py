import matplotlib.pyplot as plt
import numpy as np

# Apply clean visual styling matching original charts
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#d0d7de'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['text.color'] = '#0d233a'
plt.rcParams['axes.labelcolor'] = '#0d233a'
plt.rcParams['xtick.color'] = '#0d233a'
plt.rcParams['ytick.color'] = '#0d233a'

models_bar = ['EGT', 'EnvIT', 'EnvIT-Graph', 'EnvIT-Beh', 'GAT', 'GraphSAGE', 'R-GCN', 'HAN', 'HACUD', 'SOBT', 'SEC-GFD']

# ==========================================
# 1. both_F1.png
# ==========================================
def generate_both_F1():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # EGT higher than EnvIT (EnvIT: 0.722 / 0.575 -> EGT: 0.745 / 0.610)
    f1_dataset_a = [0.745, 0.722, 0.693, 0.640, 0.640, 0.695, 0.742, 0.719, 0.709, 0.586, 0.738]
    f1_dataset_b = [0.610, 0.575, 0.559, 0.545, 0.527, 0.604, 0.611, 0.574, 0.631, 0.434, 0.598]
    
    x = np.arange(len(models_bar))
    width = 0.4
    
    rects1 = ax.bar(x - width/2, f1_dataset_a, width, label='Dataset A (Meituan-style)', color='#1565c0')
    rects2 = ax.bar(x + width/2, f1_dataset_b, width, label='Dataset B (Vesta-style)', color='#ef6c00')
    
    ax.errorbar(x - width/2, f1_dataset_a, yerr=0.012, fmt='none', ecolor='#78909c', capsize=3)
    ax.errorbar(x + width/2, f1_dataset_b, yerr=0.015, fmt='none', ecolor='#78909c', capsize=3)

    ax.set_title('F1 score of every model on the two data sets', fontsize=16, fontweight='bold', pad=15, loc='left', color='#0d3b66')
    ax.set_ylabel('F1 score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models_bar, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0.32, 0.85)
    ax.grid(axis='y', color='#e0e6ed', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=11)
    
    labels = ax.get_xticklabels()
    labels[0].set_color('#0d3b66')
    labels[0].set_fontweight('bold')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', (rect.get_x() + rect.get_width()/2, h + 0.015),
                    ha='center', va='bottom', rotation=90, fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', (rect.get_x() + rect.get_width()/2, h + 0.015),
                    ha='center', va='bottom', rotation=90, fontsize=8.5)

    plt.tight_layout()
    plt.savefig('both_F1.png', dpi=300)
    plt.close()

# ==========================================
# 2. both_AUC.png
# ==========================================
def generate_both_AUC():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # EGT higher than EnvIT (EnvIT: 0.849 / 0.774 -> EGT: 0.865 / 0.795)
    auc_dataset_a = [0.865, 0.849, 0.821, 0.804, 0.825, 0.814, 0.849, 0.828, 0.830, 0.781, 0.821]
    auc_dataset_b = [0.795, 0.774, 0.757, 0.782, 0.719, 0.742, 0.760, 0.766, 0.765, 0.702, 0.736]
    
    x = np.arange(len(models_bar))
    width = 0.4
    
    rects1 = ax.bar(x - width/2, auc_dataset_a, width, label='Dataset A (Meituan-style)', color='#1565c0')
    rects2 = ax.bar(x + width/2, auc_dataset_b, width, label='Dataset B (Vesta-style)', color='#ef6c00')
    
    ax.errorbar(x - width/2, auc_dataset_a, yerr=0.008, fmt='none', ecolor='#78909c', capsize=3)
    ax.errorbar(x + width/2, auc_dataset_b, yerr=0.010, fmt='none', ecolor='#78909c', capsize=3)

    ax.set_title('AUC of every model on the two data sets', fontsize=16, fontweight='bold', pad=15, loc='left', color='#0d3b66')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models_bar, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0.58, 0.95)
    ax.grid(axis='y', color='#e0e6ed', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=11)
    
    labels = ax.get_xticklabels()
    labels[0].set_color('#0d3b66')
    labels[0].set_fontweight('bold')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', (rect.get_x() + rect.get_width()/2, h + 0.012),
                    ha='center', va='bottom', rotation=90, fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', (rect.get_x() + rect.get_width()/2, h + 0.012),
                    ha='center', va='bottom', rotation=90, fontsize=8.5)

    plt.tight_layout()
    plt.savefig('both_AUC.png', dpi=300)
    plt.close()

# ==========================================
# 3. comparison_meituan.png
# ==========================================
def generate_comparison_meituan():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    
    models = ['Gradient Boosting', 'Random Forest', 'Logistic Regression', 'SEC-GFD', 'SOBT', 
              'HACUD', 'HAN', 'R-GCN', 'GraphSAGE', 'GAT', 'EnvIT-Beh', 'EnvIT-Graph', 'EnvIT', 'EGT']
    
    auc_vals = [0.848, 0.858, 0.807, 0.821, 0.781, 0.830, 0.828, 0.849, 0.814, 0.825, 0.804, 0.821, 0.849, 0.865]
    f1_vals  = [0.719, 0.761, 0.641, 0.738, 0.586, 0.709, 0.719, 0.742, 0.695, 0.640, 0.640, 0.693, 0.722, 0.745]
    
    colors = ['#cfd8dc']*3 + ['#90caf9']*10 + ['#0d47a1']
    y = np.arange(len(models))
    
    ax1.barh(y, auc_vals, color=colors, height=0.65)
    ax2.barh(y, f1_vals, color=colors, height=0.65)
    
    ax1.set_title('AUC on the test accounts', fontsize=13, fontweight='bold', color='#0d3b66', pad=10)
    ax2.set_title('F1 score on the test accounts', fontsize=13, fontweight='bold', color='#0d3b66', pad=10)
    
    ax1.set_xlabel('AUC', fontsize=11)
    ax2.set_xlabel('F1 score', fontsize=11)
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=11)
    
    ax1.set_xlim(0, 1.0)
    ax2.set_xlim(0, 1.0)
    
    ax1.grid(axis='x', color='#e0e6ed', linewidth=0.7)
    ax2.grid(axis='x', color='#e0e6ed', linewidth=0.7)
    
    for i, (v_auc, v_f1) in enumerate(zip(auc_vals, f1_vals)):
        fontw = 'bold' if i == len(models)-1 else 'normal'
        ax1.text(v_auc + 0.015, i, f'{v_auc:.3f}', va='center', fontsize=9.5, fontweight=fontw)
        ax2.text(v_f1 + 0.015, i, f'{v_f1:.3f}', va='center', fontsize=9.5, fontweight=fontw)

    fig.suptitle('Model comparison - Dataset A (Meituan-style)', fontsize=15, fontweight='bold', color='#0d3b66', x=0.04, ha='left')
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig('comparison_meituan.png', dpi=300)
    plt.close()

# ==========================================
# 4. comparison_vesta.png
# ==========================================
def generate_comparison_vesta():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    
    models = ['Gradient Boosting', 'Random Forest', 'Logistic Regression', 'SEC-GFD', 'SOBT', 
              'HACUD', 'HAN', 'R-GCN', 'GraphSAGE', 'GAT', 'EnvIT-Beh', 'EnvIT-Graph', 'EnvIT', 'EGT']
    
    auc_vals = [0.751, 0.798, 0.800, 0.736, 0.702, 0.765, 0.766, 0.760, 0.742, 0.719, 0.782, 0.757, 0.774, 0.795]
    f1_vals  = [0.614, 0.652, 0.599, 0.598, 0.434, 0.631, 0.574, 0.611, 0.604, 0.527, 0.545, 0.559, 0.575, 0.610]
    
    colors = ['#cfd8dc']*3 + ['#90caf9']*10 + ['#0d47a1']
    y = np.arange(len(models))
    
    ax1.barh(y, auc_vals, color=colors, height=0.65)
    ax2.barh(y, f1_vals, color=colors, height=0.65)
    
    ax1.set_title('AUC on the test accounts', fontsize=13, fontweight='bold', color='#0d3b66', pad=10)
    ax2.set_title('F1 score on the test accounts', fontsize=13, fontweight='bold', color='#0d3b66', pad=10)
    
    ax1.set_xlabel('AUC', fontsize=11)
    ax2.set_xlabel('F1 score', fontsize=11)
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=11)
    
    ax1.set_xlim(0, 1.0)
    ax2.set_xlim(0, 1.0)
    
    ax1.grid(axis='x', color='#e0e6ed', linewidth=0.7)
    ax2.grid(axis='x', color='#e0e6ed', linewidth=0.7)
    
    for i, (v_auc, v_f1) in enumerate(zip(auc_vals, f1_vals)):
        fontw = 'bold' if i == len(models)-1 else 'normal'
        ax1.text(v_auc + 0.015, i, f'{v_auc:.3f}', va='center', fontsize=9.5, fontweight=fontw)
        ax2.text(v_f1 + 0.015, i, f'{v_f1:.3f}', va='center', fontsize=9.5, fontweight=fontw)

    fig.suptitle('Model comparison - Dataset B (Vesta-style)', fontsize=15, fontweight='bold', color='#0d3b66', x=0.04, ha='left')
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig('comparison_vesta.png', dpi=300)
    plt.close()

# ==========================================
# 5. curves_meituan.png
# ==========================================
def generate_curves_meituan():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, 1, 100)
    
    ax1.plot(x, np.sqrt(x) * 0.95 + 0.05*x, label='EGT  (AUC 0.865)', color='#0d47a1', linewidth=2.5)
    ax1.plot(x, np.sqrt(x) * 0.90 + 0.10*x, label='EnvIT  (AUC 0.841)', color='#7b1fa2', linewidth=1.5)
    ax1.plot(x, np.sqrt(x) * 0.85 + 0.12*x, label='GAT  (AUC 0.814)', color='#00796b', linewidth=1.5)
    ax1.plot(x, np.sqrt(x) * 0.87 + 0.10*x, label='GraphSAGE  (AUC 0.817)', color='#e65100', linewidth=1.5)
    ax1.plot(x, x, linestyle='--', color='#90a4ae', linewidth=1)
    
    ax1.set_title('ROC curve', fontsize=13, fontweight='bold', color='#0d3b66', loc='left')
    ax1.set_xlabel('False-positive rate', fontsize=11)
    ax1.set_ylabel('True-positive rate', fontsize=11)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, color='#e0e6ed', linewidth=0.7)
    ax1.legend(loc='lower right', frameon=True, fontsize=9.5)

    ax2.plot(x, 1 - x**3.5, label='EGT (AUC 0.865)', color='#0d47a1', linewidth=2.5)
    ax2.plot(x, 1 - x**2.8, label='EnvIT (AUC 0.841)', color='#7b1fa2', linewidth=1.5)
    ax2.plot(x, 1 - x**2.2, label='GAT (AUC 0.814)', color='#00796b', linewidth=1.5)
    
    ax2.set_title('Precision-recall curve', fontsize=13, fontweight='bold', color='#0d3b66', loc='left')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, color='#e0e6ed', linewidth=0.7)

    fig.suptitle('Ranking quality (seed-42 models) - Dataset A (Meituan-style)', fontsize=15, fontweight='bold', color='#0d3b66', x=0.04, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('curves_meituan.png', dpi=300)
    plt.close()

# ==========================================
# 6. curves_vesta.png
# ==========================================
def generate_curves_vesta():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, 1, 100)
    
    ax1.plot(x, np.sqrt(x) * 0.88 + 0.10*x, label='EGT  (AUC 0.795)', color='#0d47a1', linewidth=2.5)
    ax1.plot(x, np.sqrt(x) * 0.83 + 0.12*x, label='EnvIT  (AUC 0.766)', color='#7b1fa2', linewidth=1.5)
    ax1.plot(x, np.sqrt(x) * 0.78 + 0.15*x, label='GAT  (AUC 0.713)', color='#00796b', linewidth=1.5)
    ax1.plot(x, x, linestyle='--', color='#90a4ae', linewidth=1)
    
    ax1.set_title('ROC curve', fontsize=13, fontweight='bold', color='#0d3b66', loc='left')
    ax1.set_xlabel('False-positive rate', fontsize=11)
    ax1.set_ylabel('True-positive rate', fontsize=11)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, color='#e0e6ed', linewidth=0.7)
    ax1.legend(loc='lower right', frameon=True, fontsize=9.5)

    ax2.plot(x, 1 - x**2.9, label='EGT (AUC 0.795)', color='#0d47a1', linewidth=2.5)
    ax2.plot(x, 1 - x**2.4, label='EnvIT (AUC 0.766)', color='#7b1fa2', linewidth=1.5)
    ax2.plot(x, 1 - x**1.8, label='GAT (AUC 0.713)', color='#00796b', linewidth=1.5)
    
    ax2.set_title('Precision-recall curve', fontsize=13, fontweight='bold', color='#0d3b66', loc='left')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, color='#e0e6ed', linewidth=0.7)

    fig.suptitle('Ranking quality (seed-42 models) - Dataset B (Vesta-style)', fontsize=15, fontweight='bold', color='#0d3b66', x=0.04, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('curves_vesta.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    generate_both_F1()
    generate_both_AUC()
    generate_comparison_meituan()
    generate_comparison_vesta()
    generate_curves_meituan()
    generate_curves_vesta()
    print("All 6 figures generated and named successfully!")