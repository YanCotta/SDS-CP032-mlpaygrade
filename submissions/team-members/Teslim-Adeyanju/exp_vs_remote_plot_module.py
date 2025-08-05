"""
Experience Level vs Remote Work Analysis Module

This module provides functions to analyze and visualize the relationship between
employee experience levels and remote work arrangements.
"""

import pandas as pd
import matplotlib.pyplot as plt


def print_analysis_summary(crosstab_data):
    """Print basic analysis summary of the experience vs remote work data."""
    print("Experience Level vs Remote Work Ratio Analysis")
    print("=" * 55)
    print(crosstab_data)


def create_comprehensive_visualization(crosstab_data, figsize=(16, 12)):
    """
    Create a comprehensive 4-panel visualization of experience vs remote work data.
    
    Args:
        crosstab_data (pd.DataFrame): Crosstab of experience levels vs work arrangements
        figsize (tuple): Figure size for the plot
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    # Calculate metrics
    row_totals = crosstab_data.sum(axis=1)
    row_percentages = crosstab_data.div(row_totals, axis=0) * 100
    
    # Color scheme
    colors = ['#E74C3C', '#F39C12', '#27AE60']  # Red=Onsite, Orange=Hybrid, Green=Remote
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Experience Level vs Remote Work Arrangements - Key Insights', 
                 fontsize=16, fontweight='bold')
    
    # Panel 1: Stacked bar chart - absolute numbers
    _create_absolute_numbers_chart(crosstab_data, ax1, colors, row_totals)
    
    # Panel 2: Percentage stacked bar
    _create_percentage_chart(row_percentages, ax2, colors)
    
    # Panel 3: Remote vs non-remote focus
    _create_remote_focus_chart(crosstab_data, ax3)
    
    # Panel 4: Hybrid gap analysis
    _create_hybrid_gap_chart(crosstab_data, ax4, row_totals)
    
    plt.tight_layout()
    return fig, ((ax1, ax2), (ax3, ax4))


def _create_absolute_numbers_chart(data, ax, colors, row_totals):
    """Create the absolute numbers stacked bar chart."""
    data.plot(kind='bar', stacked=True, ax=ax, color=colors,
              edgecolor='black', linewidth=0.5)
    
    ax.set_title('Work Arrangement Distribution by Experience Level', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Experience Level')
    ax.set_ylabel('Number of Employees')
    ax.legend(title='Work Arrangement', loc='upper left')
    ax.tick_params(axis='x', rotation=90)
    
    # Add labels
    for i, (idx, row) in enumerate(data.iterrows()):
        total = row.sum()
        remote_pct = (row['Remote'] / total) * 100
        ax.text(i, total + 150, f'{total:,}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
        ax.text(i, row['Onsite'] + row['Hybrid'] + row['Remote']/2, 
                f'{remote_pct:.0f}%\nRemote', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=9)


def _create_percentage_chart(row_percentages, ax, colors):
    """Create the percentage stacked bar chart."""
    row_percentages.plot(kind='bar', stacked=True, ax=ax, color=colors)
    
    ax.set_title('Work Arrangement as % of Each Experience Level', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Experience Level')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Work Arrangement', loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='x', rotation=90)
    
    # Add percentage labels for remote work
    for i, (idx, row) in enumerate(row_percentages.iterrows()):
        remote_pct = row['Remote']
        hybrid_pct = row['Hybrid']
        ax.text(i, row['Onsite'] + hybrid_pct + remote_pct/2, f'{remote_pct:.0f}%', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=10)


def _create_remote_focus_chart(data, ax):
    """Create the remote vs non-remote focused chart."""
    remote_data = pd.DataFrame({
        'Remote': data['Remote'],
        'Non-Remote (Onsite + Hybrid)': data['Onsite'] + data['Hybrid']
    })
    
    remote_data.plot(kind='bar', ax=ax, color=['#27AE60', '#34495E'], width=0.7)
    
    ax.set_title('Remote vs Non-Remote Work by Experience Level', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Experience Level')
    ax.set_ylabel('Number of Employees')
    ax.legend(title='Work Type')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y', alpha=0.3)
    
    # Add remote adoption percentage labels
    for i, (idx, row) in enumerate(data.iterrows()):
        total = row.sum()
        remote_pct = (row['Remote'] / total) * 100
        ax.text(i, row['Remote'] + 100, f'{remote_pct:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color='#27AE60')


def _create_hybrid_gap_chart(data, ax, row_totals):
    """Create the hybrid gap analysis chart."""
    hybrid_pct_by_level = (data['Hybrid'] / row_totals * 100)
    remote_pct_by_level = (data['Remote'] / row_totals * 100)
    
    x = range(len(data.index))
    ax.bar(x, hybrid_pct_by_level, color='#F39C12', alpha=0.8, 
           label='Hybrid %', width=0.4)
    ax.bar([i+0.4 for i in x], remote_pct_by_level, color='#27AE60', alpha=0.8, 
           label='Remote %', width=0.4)
    
    ax.set_title('Hybrid vs Remote Adoption Rates\n(The "Hybrid Gap" Phenomenon)', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Experience Level')
    ax.set_ylabel('Percentage of Experience Level')
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(data.index, rotation=90)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, pct in enumerate(hybrid_pct_by_level):
        ax.text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    for i, pct in enumerate(remote_pct_by_level):
        ax.text(i+0.4, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')


def analyze_experience_vs_remote(crosstab_data, show_summary=True, show_plot=True, figsize=(16, 12)):
    """
    Main function to analyze experience level vs remote work arrangements.
    
    Args:
        crosstab_data (pd.DataFrame): Crosstab of experience levels vs work arrangements
        show_summary (bool): Whether to print analysis summary
        show_plot (bool): Whether to display the visualization
        figsize (tuple): Figure size for the plot
    """
    if show_summary:
        print_analysis_summary(crosstab_data)
    
    if show_plot:
        fig, axes = create_comprehensive_visualization(crosstab_data, figsize)
        plt.show()
        plt.close()  # Close the figure to free memory


# Example usage:
# if __name__ == "__main__":
#     # Replace 'exp_vs_remote' with your actual crosstab variable name
#     analyze_experience_vs_remote(exp_vs_remote)