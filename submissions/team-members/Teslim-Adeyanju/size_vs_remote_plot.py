"""
Module: size_vs_remote_plot

Purpose:
    Visualize company size vs remote work arrangements using various charts.
    
Function:
    plot_analysis(df: pd.DataFrame)
    
Usage:
    from size_vs_remote_plot import plot_analysis
    plot_analysis(size_vs_remote)

Input Data:
    The input DataFrame must have the following format (index = company size):

                    Onsite  Hybrid  Remote
    company_size                            
    Large              603     152     283
    Medium           10440      58    4770
    Small               41      39     108
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_analysis(size_vs_remote: pd.DataFrame):
    """
    Generate comprehensive visual analysis of company size vs remote work arrangements.

    Parameters:
    -----------
    size_vs_remote : pd.DataFrame
        DataFrame with index as company sizes and columns: ['Onsite', 'Hybrid', 'Remote']
    """

    # Calculate key metrics
    row_totals = size_vs_remote.sum(axis=1)
    col_totals = size_vs_remote.sum(axis=0)
    total_employees = row_totals.sum()
    row_percentages = size_vs_remote.div(row_totals, axis=0) * 100

    # Create figure and axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Company Size vs Remote Work Arrangements - Key Insights', fontsize=16, fontweight='bold')

    # 1. Stacked Bar Chart - Absolute Numbers
    size_vs_remote.plot(kind='bar', stacked=True, ax=ax1, 
                        color=['#E74C3C', '#F39C12', '#27AE60'],
                        edgecolor='black', linewidth=0.5)
    ax1.set_title('Work Arrangement Distribution by Company Size', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Company Size')
    ax1.set_ylabel('Number of Employees')
    ax1.legend(title='Work Arrangement', loc='upper left')
    ax1.tick_params(axis='x', rotation=0)

    for i, (idx, row) in enumerate(size_vs_remote.iterrows()):
        total = row.sum()
        remote_pct = (row['Remote'] / total) * 100
        ax1.text(i, total + 300, f'{total:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.text(i, row['Onsite'] + row['Hybrid'] + row['Remote'] / 2,
                 f'{remote_pct:.0f}%\nRemote', ha='center', va='center',
                 fontweight='bold', color='white', fontsize=9)

    # 2. Percentage Stacked Bar - Row Percentage
    row_percentages.plot(kind='bar', stacked=True, ax=ax2,
                         color=['#E74C3C', '#F39C12', '#27AE60'])
    ax2.set_title('Work Arrangement as % of Each Company Size', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Company Size')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(title='Work Arrangement', loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.tick_params(axis='x', rotation=0)

    for i, (idx, row) in enumerate(row_percentages.iterrows()):
        remote_pct = row['Remote']
        hybrid_pct = row['Hybrid']
        ax2.text(i, row['Onsite'] + hybrid_pct + remote_pct / 2,
                 f'{remote_pct:.0f}%', ha='center', va='center',
                 fontweight='bold', color='white', fontsize=10)

    # 3. Remote vs Non-Remote
    remote_data = pd.DataFrame({
        'Remote': size_vs_remote['Remote'],
        'Non-Remote (Onsite + Hybrid)': size_vs_remote['Onsite'] + size_vs_remote['Hybrid']
    })

    remote_data.plot(kind='bar', ax=ax3, color=['#27AE60', '#34495E'], width=0.7)
    ax3.set_title('Remote vs Non-Remote Work by Company Size', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Company Size')
    ax3.set_ylabel('Number of Employees')
    ax3.legend(title='Work Type')
    ax3.tick_params(axis='x', rotation=0)
    ax3.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(size_vs_remote.iterrows()):
        total = row.sum()
        remote_pct = (row['Remote'] / total) * 100
        ax3.text(i, row['Remote'] + 200, f'{remote_pct:.1f}%',
                 ha='center', va='bottom', fontweight='bold', fontsize=9, color='#27AE60')

    # 4. Hybrid vs Remote % (Gap)
    hybrid_pct_by_size = (size_vs_remote['Hybrid'] / row_totals * 100)
    remote_pct_by_size = (size_vs_remote['Remote'] / row_totals * 100)

    x = range(len(size_vs_remote.index))
    ax4.bar(x, hybrid_pct_by_size, color='#F39C12', alpha=0.8, label='Hybrid %', width=0.4)
    ax4.bar([i + 0.4 for i in x], remote_pct_by_size, color='#27AE60', alpha=0.8, label='Remote %', width=0.4)

    ax4.set_title('Hybrid vs Remote Adoption Rates\n(The "Hybrid Gap" by Company Size)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Company Size')
    ax4.set_ylabel('Percentage of Company Size')
    ax4.set_xticks([i + 0.2 for i in x])
    ax4.set_xticklabels(size_vs_remote.index, rotation=0)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    for i, pct in enumerate(hybrid_pct_by_size):
        ax4.text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for i, pct in enumerate(remote_pct_by_size):
        ax4.text(i + 0.4, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()
