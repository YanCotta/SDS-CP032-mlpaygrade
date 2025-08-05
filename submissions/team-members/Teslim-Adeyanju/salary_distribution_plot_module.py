"""
Module: salary_distribution_analysis

Purpose:
    Analyze salary distribution, detect outliers, and visualize salary patterns.

Function:
    analyze_salary_distribution(df: pd.DataFrame)

Usage:
    from salary_distribution_plot_module import analyze_salary_distribution
    analyze_salary_distribution(df)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean plotting style
plt.style.use('default')
sns.set_style("whitegrid", {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.3
})

def analyze_salary_distribution(df: pd.DataFrame):
    """
    Perform comprehensive analysis and visualization of salary distribution.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain a column 'salary_in_usd'.
    """

    # Key statistics
    salary_stats = df['salary_in_usd'].describe()
    mean_salary = salary_stats['mean']
    median_salary = salary_stats['50%']
    std_dev = salary_stats['std']
    q1, q3 = salary_stats['25%'], salary_stats['75%']
    iqr = q3 - q1

    # Outlier thresholds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df['salary_in_usd'] < lower_bound) | (df['salary_in_usd'] > upper_bound)]
    outlier_pct = (len(outliers) / len(df)) * 100

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1], 
                         hspace=0.3, wspace=0.3)

    # Main KDE plot (larger, cleaner)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Clean KDE plot
    sns.kdeplot(data=df, x='salary_in_usd', ax=ax1, 
                color='#2E86AB', fill=True, alpha=0.3, linewidth=2.5)
    
    ax1.set_title('💰 Salary Distribution Analysis', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Annual Salary (USD)', fontsize=14, fontweight='semibold')
    ax1.set_ylabel('Density', fontsize=14, fontweight='semibold')
    
    # Add IQR shading
    ax1.axvspan(q1, q3, color='#F18F01', alpha=0.15, label='IQR (Middle 50%)')
    
    # Clean vertical lines for key statistics
    stats_lines = [
        ('Median', median_salary, '#2ECC71', '-'),
        ('Mean', mean_salary, '#E74C3C', '--'),
        ('Q1', q1, '#F39C12', ':'),
        ('Q3', q3, '#F39C12', ':')
    ]
    
    y_max = ax1.get_ylim()[1]
    for i, (label, value, color, style) in enumerate(stats_lines):
        ax1.axvline(value, color=color, linestyle=style, linewidth=2, alpha=0.8)
        
        # Position labels to avoid overlap
        y_pos = y_max * (0.85 - i * 0.15)
        ax1.annotate(f'{label}: ${value:,.0f}', 
                    xy=(value, y_pos), xytext=(10, 0),
                    textcoords='offset points', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2),
                    fontsize=11, fontweight='bold', color=color)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)

    # Statistics panel (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = f"""
    📊 KEY STATISTICS
    ━━━━━━━━━━━━━━━━━━━
    
    Mean:     ${mean_salary:,.0f}
    Median:   ${median_salary:,.0f}
    Std Dev:  ${std_dev:,.0f}
    
    Q1 (25%): ${q1:,.0f}
    Q3 (75%): ${q3:,.0f}
    IQR:      ${iqr:,.0f}
    
    Min:      ${salary_stats['min']:,.0f}
    Max:      ${salary_stats['max']:,.0f}
    
    🚨 OUTLIERS
    ━━━━━━━━━━━━━━━━━━━
    Count:    {len(outliers):,}
    Percent:  {outlier_pct:.1f}%
    Threshold: ${upper_bound:,.0f}
    
    📈 DISTRIBUTION
    ━━━━━━━━━━━━━━━━━━━
    Skewness: {df['salary_in_usd'].skew():.2f}
    Kurtosis: {df['salary_in_usd'].kurtosis():.2f}
    """
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.8))

    # Box plot (bottom left)
    ax3 = fig.add_subplot(gs[1, :2])
    
    box_plot = ax3.boxplot(df['salary_in_usd'], vert=False, patch_artist=True,
                          boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                          medianprops=dict(color='#E74C3C', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='#C73E1D', 
                                        markeredgecolor='#C73E1D', markersize=4, alpha=0.6))
    
    ax3.set_title('📦 Box Plot - Outlier Detection', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Annual Salary (USD)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add outlier count annotation
    ax3.text(0.02, 0.8, f'Outliers: {len(outliers):,} ({outlier_pct:.1f}%)', 
             transform=ax3.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5E5', alpha=0.8))

    # Salary brackets (bottom right)
    ax4 = fig.add_subplot(gs[1, 2])
    
    bins = [0, 75000, 125000, 175000, 225000, 300000, float('inf')]
    labels = ['<75k', '75-125k', '125-175k', '175-225k', '225-300k', '300k+']
    df_temp = df.copy()
    df_temp['salary_bracket'] = pd.cut(df_temp['salary_in_usd'], bins=bins, labels=labels)
    bracket_counts = df_temp['salary_bracket'].value_counts().sort_index()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    wedges, texts, autotexts = ax4.pie(bracket_counts.values, labels=bracket_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax4.set_title('💼 Salary Brackets', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    # Percentile analysis (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [df['salary_in_usd'].quantile(p/100) for p in percentiles]
    
    bars = ax5.bar(range(len(percentiles)), percentile_values, 
                   color='#2E86AB', alpha=0.7, edgecolor='white', linewidth=1)
    
    ax5.set_title('📊 Salary Percentiles', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Percentile', fontsize=12)
    ax5.set_ylabel('Salary (USD)', fontsize=12)
    ax5.set_xticks(range(len(percentiles)))
    ax5.set_xticklabels([f'{p}%' for p in percentiles])
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, percentile_values)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'${value:,.0f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

    plt.suptitle('🎯 Comprehensive Salary Distribution Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()

    # Enhanced summary
    print("\n" + "="*60)
    print("💰 SALARY DISTRIBUTION INSIGHTS")
    print("="*60)
    
    print(f"\n📈 CENTRAL TENDENCY:")
    print(f"   Mean:   ${mean_salary:>12,.0f}")
    print(f"   Median: ${median_salary:>12,.0f} (${mean_salary-median_salary:+,.0f} difference)")
    
    print(f"\n📊 VARIABILITY:")
    print(f"   Range:  ${salary_stats['min']:>12,.0f} - ${salary_stats['max']:,.0f}")
    print(f"   IQR:    ${q1:>12,.0f} - ${q3:,.0f}")
    print(f"   Std:    ${std_dev:>12,.0f}")
    
    print(f"\n🚨 OUTLIERS:")
    print(f"   Count:     {len(outliers):>8,} ({outlier_pct:.1f}% of data)")
    print(f"   Threshold: ${upper_bound:>12,.0f}")
    
    print(f"\n🎯 KEY PERCENTILES:")
    for p in [10, 25, 50, 75, 90, 95]:
        value = df['salary_in_usd'].quantile(p/100)
        print(f"   {p:>2}%:     ${value:>12,.0f}")
    
    print("\n" + "="*60)
