import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full", app_title="salary_prediction")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    return np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv('data.csv',  engine="pyarrow")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()

    return


@app.cell
def _(df):
    numerical_data = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_data = df.select_dtypes(include=['object', 'category']).columns

    # keep the columns to be able to use the data in ploting and subsiting use. 
    return categorical_data, numerical_data


@app.cell
def _(categorical_data):
    categorical_data.to_list() 
    return


@app.cell
def _(numerical_data):
    numerical_data
    return


@app.cell
def _(categorical_data, df):
    # Loop through columns
    for i in categorical_data:
  
      # Print the number of unique values
      print(f"Number of unique values in {i} column: ", df[i].nunique())

    return


@app.cell
def _(df, numerical_data, plt, sns):
    # plotting the kde for the numerical columns
    for col in numerical_data:
        plt.figure(figsize=(15,6), dpi=100, facecolor='w', edgecolor='k')
        sns.kdeplot(df[col], cut=0.5)  
        plt.title(f'{col} distribution')
        plt.show()


    return


@app.cell
def _(df, numerical_data, plt):
    # plotting the boxplot for the numerical columns
    df[numerical_data].boxplot(figsize=(15,10), color='black')
    plt.suptitle('Boxplots of Numerical Columns', x=0.5, y=1.02, ha='center', fontsize='large')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, numerical_data, plt):
    # plotting the boxplot for the numerical columns
    fig, axes = plt.subplots(1, len(numerical_data), figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')
    axes = axes.flatten()
    plt.suptitle('Boxplots of Numerical Columns', x=0.5, y=1.02, ha='center', fontsize='large')
    for m, column in enumerate(numerical_data):
        df[column].plot(kind='box', ax=axes[m])
        axes[m].set_title(column)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, numerical_data):
    # Calculate correlation matrix
    correlation = df[numerical_data].corr()
    correlation
    return (correlation,)


@app.cell
def _(correlation, np, plt, sns):
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    # Plot the heatmap with the mask
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation, 
        mask=mask,               
        annot=True,              
        cmap='coolwarm',         
        linewidths=0.5,          
        vmin=-1, vmax=1,         
        fmt=".2f",               # Format annotations to 2 decimal places
        annot_kws={"size": 8}    # Adjust annotation text size
    )
    plt.title('Numerical Features Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.ui.datetime()
    return


@app.cell
def _(md, mo, plt):


    plt.plot([1, 2])
    axis = plt.gca()
    md(f"Here's a plot: {mo.as_html(axis)}")
    return


if __name__ == "__main__":
    app.run()
