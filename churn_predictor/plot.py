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
