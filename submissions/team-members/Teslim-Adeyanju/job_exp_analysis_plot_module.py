# job_exp_analysis.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_job_vs_experience(job_vs_exp):
    print("Top 10 Job Titles vs Experience Level Analysis")
    print("=" * 60)
    print(job_vs_exp)

    # Calculate metrics
    row_totals = job_vs_exp.sum(axis=1).sort_values(ascending=False)
    job_vs_exp_sorted = job_vs_exp.loc[row_totals.index]

    # Set figure
    fig = plt.figure(figsize=(20, 16))

    # 1. Main Stacked Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    job_vs_exp_sorted.plot(kind='bar', stacked=True, ax=ax1, 
                           color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60'],
                           edgecolor='black', linewidth=0.5, width=0.8)
    ax1.set_title('Job Titles by Experience Level Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Job Title')
    ax1.set_ylabel('Number of Positions')
    ax1.legend(title='Experience Level', loc='upper right')
    ax1.tick_params(axis='x', rotation=90)
    ax1.grid(axis='y', alpha=0.3)

    # Total labels
    for i, total in enumerate(row_totals):
        ax1.text(i, total + 50, f'{total:,}', ha='center', fontweight='bold', fontsize=9)

    # 2. Senior Level Positions
    senior_data = job_vs_exp_sorted['Senior Level'].sort_values()
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.barh(senior_data.index, senior_data.values, color='#27AE60', alpha=0.8)
    ax2.set_title('Senior Level Positions by Job Title', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Senior Positions')
    ax2.grid(axis='x', alpha=0.3)

    for bar in bars2:
        ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, 
                 f'{int(bar.get_width()):,}', va='center', fontsize=9, fontweight='bold')

    # 3. Entry Level Positions
    entry_data = job_vs_exp_sorted['Entry Level'].sort_values(ascending=False)
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(entry_data.index, entry_data.values, color='#3498DB', alpha=0.8)
    ax3.set_title('Entry Level Opportunities by Job Title', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Entry Positions')
    ax3.tick_params(axis='x', rotation=90)
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars3, entry_data):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val}', ha='center', fontsize=8, fontweight='bold')

    # 4. Career Progression Ratios (Senior/Entry)
    ratios = job_vs_exp_sorted.apply(
        lambda row: row['Senior Level'] / row['Entry Level'] if row['Entry Level'] > 0 else np.nan, axis=1
    ).dropna().sort_values()

    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.barh(ratios.index, ratios.values, color='#9B59B6', alpha=0.8)
    ax4.axvline(1, color='red', linestyle='--', alpha=0.7, label='1:1 Ratio')
    ax4.set_title('Career Progression Ratio (Senior/Entry)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Senior to Entry Ratio')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars4, ratios):
        ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}:1', va='center', fontsize=8, fontweight='bold')

    # 5. Executive Level Representation Pie Chart
    exec_data = job_vs_exp_sorted['Executive Level']
    exec_data_nonzero = exec_data[exec_data > 0]

    ax5 = plt.subplot(2, 3, 5)
    colors_exec = plt.cm.Set3(np.linspace(0, 1, len(exec_data_nonzero)))
    ax5.pie(exec_data_nonzero, labels=exec_data_nonzero.index, autopct='%1.0f%%', 
            colors=colors_exec, startangle=90, textprops={'fontsize':8, 'color':'white', 'weight':'bold'})
    ax5.set_title('Executive Level Distribution', fontsize=12, fontweight='bold')

    # 6. Experience Level Percentage Heatmap
    ax6 = plt.subplot(2, 3, 6)
    percentages = job_vs_exp_sorted.div(job_vs_exp_sorted.sum(axis=1), axis=0) * 100
    sns.heatmap(percentages, annot=True, fmt='.0f', cmap='RdYlGn_r', cbar_kws={'label':'%'}, ax=ax6)
    ax6.set_title('Experience Distribution (%)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Experience Level')
    ax6.set_ylabel('Job Title')
    ax6.tick_params(axis='x', rotation=90)
    ax6.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()
