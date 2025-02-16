import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_horizontal_boxplot(df, column_name):
    """
    Plot a horizontal boxplot for a specified column in a DataFrame.
    
    If a list of column names is provided, plots multiple boxplots stacked vertically.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str or list): The name of the column to plot, or a list of columns.
    
    Raises:
        ValueError: If any of the specified columns are not found in the DataFrame.
    """
    # If the input is a list, plot multiple boxplots stacked vertically
    if isinstance(column_name, list):
        # Check that all columns in the list are present in the DataFrame
        missing_cols = [col for col in column_name if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")
        
        # Create a grid to plot multiple boxplots
        plt.figure(figsize=(10, 2 * len(column_name)))
        
        for i, col in enumerate(column_name, 1):
            plt.subplot(len(column_name), 1, i)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot for {col}')
            plt.xlabel(col)
        
    # If it's a single column, plot the boxplot normally
    elif column_name in df.columns:
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=df[column_name])
        plt.title(f'Boxplot for {column_name}')
        plt.xlabel(column_name)
    
    else:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Usage: plot_horizontal_boxplot(df, ['column1', 'column2'])

