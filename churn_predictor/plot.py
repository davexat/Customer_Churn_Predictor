import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

def init():
    sns.set_style('whitegrid')

def summary_cat_var(df, column, gbe=False, sbs=False, summ=True):
    df_gb = plot_barplot(df, column, gbe, sbs)
    if summ: print(df_gb)

def plot_barplot(df, column, gbe=False, sbs=False):
    if isinstance(column, list):
        for col in column: plot_barplot(df, col, gbe, sbs)
        return
    df_gb = df.groupby('Exited') if gbe else df
    df_gb = df_gb[column].value_counts(normalize=True)
    df_freq = df_gb.reset_index()

    sns.catplot(
        data=df_freq, 
        kind='bar', x=column, 
        y='proportion',
        palette = {0: '#1f77b4', 1: '#d62728'} if gbe else None,
        hue = 'Exited' if gbe else None,
        col = 'Exited' if sbs and gbe else None,
        col_order = [0,1] if sbs and gbe else None,
        height=5, 
        aspect=1 if sbs else 2,
        edgecolor='black'
    )
    plt.show()
    return df_gb

def summary_num_var(df, column, gbe=False, sbs=False, summ=True, bins=None):
    plot_histogram(df, column, gbe, sbs)
    plot_boxplot(df, column, gbe, sbs)
    info = df.groupby('Exited') if gbe else df
    if summ:
        print(info[column].describe())

def plot_boxplot(df, column, gbe=False, sbs=False):
    if isinstance(column, list):
        for col in column: plot_boxplot(df, col, gbe, sbs)
        return
    sns.catplot(
        data=df, 
        kind='box', x=column,
        palette = {0: '#1f77b4', 1: '#d62728'} if gbe else None,
        hue = 'Exited' if gbe else None, 
        col = 'Exited' if sbs and gbe else None, 
        col_order=[0,1] if sbs and gbe else None, 
        height=2, 
        aspect= 2.5 if sbs else 5
    )
    plt.show()

def plot_histogram(df, column, gbe=False, sbs=False):
    if isinstance(column, list):
        for col in column: plot_histogram(df, col, gbe, sbs)
        return
    sns.displot(
        data=df,
        kind='hist',
        x=column,
        palette={0: "#1f77b4", 1: "#d62728"} if gbe else None,
        hue='Exited' if gbe else None,
        col='Exited' if sbs and gbe else None,
        col_order=[0,1] if sbs and gbe else None,
        stat="probability",
        common_norm=False,
        kde=True,
        height=5, 
        aspect=1 if sbs else 2,
        edgecolor='black'
    )
    plt.show()

def plot_correlation_heatmap(df, method='pearson'):
    corr = df.corr(method)
    n = len(corr.columns)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr.iloc[1:, :-1], mask=mask[1:, :-1], annot=True, cmap='coolwarm', 
                vmin=-1, vmax=1, fmt='.2f', linewidths=0.5, square=True, cbar_kws={"shrink": 0.75})
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.xlabel('Variables (1 to N-1)')
    plt.ylabel('Variables (N to 2)')
    plt.show()

def analyze_categorical_relationship(df, var1, var2):
    crosstab = pd.crosstab(df[var1], df[var2], normalize='index')
    crosstab.plot(kind='bar', edgecolor='black')
    plt.title(f"Proportion of {var1} by {var2}")
    plt.ylabel("Proportion")
    plt.show()
    chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df[var1], df[var2]))
    print(crosstab)
    print(f"\nChi-square test between {var1} and {var2}:")
    print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}\n")
