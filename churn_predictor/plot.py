import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
        aspect=1 if sbs else 2
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
        aspect=1 if sbs else 2
    )
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plot a customized correlation heatmap showing only unique correlations
    based on a non-redundant triangular approach.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to compute correlations.
    
    Returns:
        None
    """
    # Calculate correlation matrix
    corr = df.corr()

    # Get the number of columns
    n = len(corr.columns)

    # Create a mask for the upper triangular matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(6, 6))

    # Plot the heatmap
    sns.heatmap(corr.iloc[1:, :-1], mask=mask[1:, :-1], annot=True, cmap='coolwarm', 
                vmin=-1, vmax=1, fmt='.2f', linewidths=0.5, square=True, cbar_kws={"shrink": 0.75})

    # Add titles and labels
    plt.title('Correlation Heatmap')
    plt.xlabel('Variables (1 to N-1)')
    plt.ylabel('Variables (N to 2)')
    
    # Display the plot
    plt.show()

# Usage example: plot_custom_correlation_heatmap(df_cleaned)

def plot_pairplot(df, columns=None, hue=None, corner=False):
    """
    Plot a pairplot to visualize relationships between numerical variables using scatterplots
    for each combination of columns. Supports both numerical and categorical hue variables.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    columns (list, optional): List of column names to include in the pairplot.
                            If None, all numerical columns will be used.
    hue (str, optional): Column name to use for color-coding points.
                        Can be either numerical or categorical.
    
    Returns:
    None
    """
    # Filter the DataFrame to include only the specified columns or default to all numerical columns
    if columns:
        plot_df = df[columns].copy()
    else:
        plot_df = df.select_dtypes(include=['number']).copy()
    
    # Handle hue parameter
    if hue:
        if hue not in df.columns:
            raise ValueError(f"Hue column '{hue}' not found in DataFrame")
        
        # Add hue column to plot_df if it's not already included
        if hue not in plot_df.columns:
            plot_df[hue] = df[hue]
        
        # Handle categorical hue
        if df[hue].dtype == 'object' or pd.api.types.is_categorical_dtype(df[hue]):
            # Convert categorical hue to numeric codes for proper plotting
            plot_df[hue] = pd.Categorical(plot_df[hue]).codes
            
            # Create custom palette and legend
            unique_categories = df[hue].unique()
            n_colors = len(unique_categories)
            palette = sns.color_palette("husl", n_colors)
            
            # Create the pairplot
            g = sns.pairplot(plot_df, hue=hue, diag_kind='kde', plot_kws={'alpha': 0.6}, corner = corner, palette=palette)
            
            # Update legend with original category names
            new_labels = [str(cat) for cat in unique_categories]
            legend = g._legend
            for t, label in zip(legend.get_texts(), new_labels):
                t.set_text(label)
        else:
            # For numerical hue, use default behavior
            g = sns.pairplot(plot_df, hue=hue, diag_kind='kde', corner = corner, plot_kws={'alpha': 0.6})
    else:
        # No hue specified
        g = sns.pairplot(plot_df, diag_kind='kde', corner = corner, plot_kws={'alpha': 0.6})
    if hue:
        g.fig.subplots_adjust(right=0.95, top=1)
    
    plt.suptitle('Pairplot of Numerical Variables', y=1.02)
    plt.show()

# Usage example: plot_pairplot(df, columns=['Age', 'Income', 'Score'], hue='Category')
