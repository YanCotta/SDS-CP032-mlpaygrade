# employment_analysis.py

import matplotlib.pyplot as plt

def plot_exp_vs_employment(exp_vs_employment):
    print("Experience Level vs Employment Type Analysis")
    print("=" * 50)
    print(exp_vs_employment)

    # Calculate key metrics
    row_totals = exp_vs_employment.sum(axis=1)
    col_totals = exp_vs_employment.sum(axis=0)
    total_employees = row_totals.sum()
    row_percentages = exp_vs_employment.div(row_totals, axis=0) * 100

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experience Level vs Employment Type - Key Insights', fontsize=16, fontweight='bold')

    # 1. Stacked Bar Chart with totals - Main insight
    exp_vs_employment.plot(kind='bar', stacked=True, ax=ax1, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                          edgecolor='black', linewidth=0.5)
    ax1.set_title('Employment Distribution by Experience Level', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Experience Level')
    ax1.set_ylabel('Number of Employees')
    ax1.legend(title='Employment Type', loc='upper left')
    ax1.tick_params(axis='x', rotation=90)  # Rotated by 90 degrees

    # Add total labels
    for i, (idx, row) in enumerate(exp_vs_employment.iterrows()):
        total = row.sum()
        ax1.text(i, total + 200, f'{total:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ft_pct = (row['Full Time'] / total) * 100
        ax1.text(i, row['Full Time']/2, f'{ft_pct:.0f}%\nFull-time', ha='center', va='center', 
                 fontweight='bold', color='white', fontsize=9)

    # 2. Percentage view to show Full-time dominance
    row_percentages.plot(kind='bar', stacked=True, ax=ax2,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax2.set_title('Employment Type as % of Each Experience Level', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Experience Level')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(title='Employment Type', loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.tick_params(axis='x', rotation=90)  # Rotated by 90 degrees
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')

    # 3. Non-Full-Time Focus (Contract + Freelance + Part-time)
    non_fulltime = exp_vs_employment.drop('Full Time', axis=1)
    non_fulltime.plot(kind='bar', ax=ax3, 
                     color=['#FF6B6B', '#4ECDC4', '#FFA07A'])
    ax3.set_title('Alternative Employment Types Only\n(Excluding Full-Time)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Experience Level')
    ax3.set_ylabel('Number of Employees')
    ax3.legend(title='Employment Type')
    ax3.tick_params(axis='x', rotation=90)  # Rotated by 90 degrees
    ax3.grid(axis='y', alpha=0.3)

    for container in ax3.containers:
        ax3.bar_label(container, fmt='%d', fontsize=9)

    # 4. Experience Level Distribution (Total workforce)
    colors_exp = ['#98D8C8', '#F7DC6F', '#BB8FCE', '#F1948A']
    wedges, texts, autotexts = ax4.pie(row_totals.values, labels=row_totals.index, 
                                      autopct='%1.1f%%', colors=colors_exp, startangle=90)
    ax4.set_title('Workforce Distribution by Experience Level', fontweight='bold', fontsize=12)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    plt.tight_layout()
    plt.show()
