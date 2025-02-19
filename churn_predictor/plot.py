import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_horizontal_boxplot(df, columns):
    """
    Plot a horizontal boxplot for a specified column in a DataFrame.
    
    If a list of column names is provided, plots multiple boxplots stacked vertically.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (str or list): The name of the column to plot, or a list of columns.
    
    Raises:
        ValueError: If any of the specified columns are not found in the DataFrame.
    """
    # If the input is a list, plot multiple boxplots stacked vertically
    if isinstance(columns, list):
        # Check that all columns in the list are present in the DataFrame
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")
        
        # Create a grid to plot multiple boxplots
        plt.figure(figsize=(6, 2 * len(columns)))
        
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot for {col}')
            plt.xlabel(col)
        
    # If it's a single column, plot the boxplot normally
    elif columns in df.columns:
        plt.figure(figsize=(6, 2))
        sns.boxplot(x=df[columns])
        plt.title(f'Boxplot for {columns}')
        plt.xlabel(columns)
    
    else:
        raise ValueError(f"Column '{columns}' not found in the DataFrame.")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Usage: plot_horizontal_boxplot(df, ['column1', 'column2'])

def plot_histograms(df, columns, bins=10, kde=False):
    """
    Plot histograms for specified columns in a DataFrame.

    If a list of column names is provided, plots multiple histograms stacked vertically.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (str or list): The name of the column to plot, or a list of columns.
        bins (int): Number of bins for the histogram (default is 10).
        kde (bool): Whether to display the kernel density estimate (KDE) curve (default is False).

    Raises:
        ValueError: If any of the specified columns are not found in the DataFrame.
    """
    if isinstance(columns, list):
        # Check that all columns in the list are present in the DataFrame
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")
        
        # Create a grid to plot multiple histograms
        plt.figure(figsize=(6, 3 * len(columns)))
        
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            sns.histplot(df[col], bins=bins, kde=kde)
            plt.title(f'Histogram for {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
    elif columns in df.columns:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[columns], bins=bins, kde=kde)
        plt.title(f'Histogram for {columns}')
        plt.xlabel(columns)
        plt.ylabel('Frequency')
    
    else:
        raise ValueError(f"Column '{columns}' not found in the DataFrame.")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Usage example: plot_histograms(df, ['column1', 'column2'], kde=True)

def plot_bar_charts(df, columns):
    """
    Plot bar charts for specified columns in a DataFrame, regardless of data type.

    If a list of column names is provided, plots multiple bar charts stacked vertically.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (str or list): The name of the column to plot, or a list of columns.

    Raises:
        ValueError: If any of the specified columns are not found in the DataFrame.
    """
    if isinstance(columns, list):
        # Check that all columns in the list are present in the DataFrame
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")
        
        # Create a grid to plot multiple bar charts
        plt.figure(figsize=(6, 3 * len(columns)))
        
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            sns.countplot(data=df, x=col)
            plt.title(f'Bar Chart for {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
    elif columns in df.columns:
        plt.figure(figsize=(6, 3))
        sns.countplot(data=df, x=columns)
        plt.title(f'Bar Chart for {columns}')
        plt.xlabel(columns)
        plt.ylabel('Frequency')
    
    else:
        raise ValueError(f"Column '{columns}' not found in the DataFrame.")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Usage example: plot_bar_charts(df, ['column1', 'column2'])

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

def plot_pairplot(df, columns=None, hue=None):
    """
    Creates a pairplot showing relationships between variables with optional color coding.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    columns (list, optional): Columns to include in the plot. Uses numerical columns if None
    hue (str, optional): Column name for color coding the points
    """
    # Select data columns
    data = df[columns].copy() if columns else df.select_dtypes(include=['number']).copy()
    
    # Add hue column if specified
    if hue:
        if hue not in df.columns:
            raise ValueError(f"Hue column '{hue}' not found in DataFrame")
        
        # Handle categorical hue with custom colors
        if df[hue].dtype == 'object' or pd.api.types.is_categorical_dtype(df[hue]):
            palette = sns.color_palette("husl", len(df[hue].unique()))
        else:
            palette = None
            
        data[hue] = df[hue]
    else:
        palette = None
    
    # Create pairplot
    g = sns.pairplot(
        data=data,
        hue=hue,
        palette=palette,
        diag_kind='hist',
        plot_kws={'alpha': 0.5}
    )
    
    # Adjust layout
    g.fig.suptitle('Pairplot of Variables', y=1.02)
    plt.tight_layout()
    
    return g

# Example usage:
# plot_pairplot(df, columns=['Age', 'Income', 'Score'], hue='Category')
    
