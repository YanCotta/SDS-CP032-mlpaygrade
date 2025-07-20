"""
Module: top_locations

Purpose:
    Visualize remote work trends by job location using various plot types.

Function:
    plot_top_location_insights(location_vs_remote: pd.DataFrame)

Usage:
    from top_locations import plot_top_location_insights
    plot_top_location_insights(location_vs_remote)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_top_location_insights(location_vs_remote: pd.DataFrame):
    """
    Visualize job distribution and remote work adoption by location.

    Parameters:
    -----------
    location_vs_remote : pd.DataFrame
        Crosstab DataFrame with index as locations and columns: ['Onsite', 'Hybrid', 'Remote']
    """

    # Set styles
    plt.style.use('default')
    sns.set_palette("husl")

    # Extract data
    locations = location_vs_remote.index.tolist()
    onsite = location_vs_remote['Onsite'].tolist()
    hybrid = location_vs_remote['Hybrid'].tolist()
    remote = location_vs_remote['Remote'].tolist()

    # Calculate totals and percentages
    totals = location_vs_remote.sum(axis=1).tolist()
    onsite_pct = (location_vs_remote['Onsite'] / location_vs_remote.sum(axis=1) * 100).tolist()
    hybrid_pct = (location_vs_remote['Hybrid'] / location_vs_remote.sum(axis=1) * 100).tolist()
    remote_pct = (location_vs_remote['Remote'] / location_vs_remote.sum(axis=1) * 100).tolist()

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    x_pos = np.arange(len(locations))
    width = 0.6

    # 1. Absolute Numbers - Stacked Bar
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.bar(x_pos, onsite, width, label='Onsite', color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos, hybrid, width, bottom=onsite, label='Hybrid', color='#A23B72', alpha=0.8)
    ax1.bar(x_pos, remote, width, bottom=np.array(onsite)+np.array(hybrid), label='Remote', color='#F18F01', alpha=0.8)
    ax1.set_xlabel('Location', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Jobs', fontsize=12, fontweight='bold')
    ax1.set_title('Job Distribution by Location (Absolute Numbers)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(locations)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for i, total in enumerate(totals):
        ax1.text(i, total + 100, f'{total:,}', ha='center', va='bottom', fontweight='bold')

    # 2. Percentage Stacked Bar Chart
    ax2 = plt.subplot(2, 3, 3)
    ax2.bar(x_pos, onsite_pct, width, label='Onsite', color='#2E86AB', alpha=0.8)
    ax2.bar(x_pos, hybrid_pct, width, bottom=onsite_pct, label='Hybrid', color='#A23B72', alpha=0.8)
    ax2.bar(x_pos, remote_pct, width, bottom=np.array(onsite_pct)+np.array(hybrid_pct), label='Remote', color='#F18F01', alpha=0.8)
    ax2.set_xlabel('Location', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Work Type Distribution by Location (%)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(locations)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Overall Work Type Distribution Pie Chart
    ax3 = plt.subplot(2, 3, 4)
    total_onsite = sum(onsite)
    total_hybrid = sum(hybrid)
    total_remote = sum(remote)
    sizes = [total_onsite, total_hybrid, total_remote]
    labels = ['Onsite', 'Hybrid', 'Remote']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax3.set_title('Overall Work Type Distribution', fontsize=14, fontweight='bold', pad=20)

    # 4. Remote Work Adoption Rate by Location
    ax4 = plt.subplot(2, 3, 5)
    colors_gradient = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(locations)))
    bars = ax4.bar(locations, remote_pct, color=colors_gradient, alpha=0.8)
    ax4.set_xlabel('Location', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Remote Work Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Remote Work Adoption by Location', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(axis='y', alpha=0.3)
    for bar, rate in zip(bars, remote_pct):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 5. Market Size by Location
    ax5 = plt.subplot(2, 3, 6)
    colors_size = plt.cm.Blues(np.linspace(0.4, 0.9, len(locations)))
    bars_size = ax5.bar(locations, totals, color=colors_size, alpha=0.8)
    ax5.set_xlabel('Location', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Total Jobs', fontsize=12, fontweight='bold')
    ax5.set_title('Market Size by Location', fontsize=14, fontweight='bold', pad=20)
    ax5.grid(axis='y', alpha=0.3)
    for bar, size in zip(bars_size, totals):
        offset = max(totals)*0.01
        rotation = 45 if size > 1000 else 0
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, 
                 f'{size:,}', ha='center', va='bottom', fontweight='bold', rotation=rotation)

    plt.tight_layout()
    plt.show()
