import pandas as pd


def convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert specific columns in the DataFrame to float type and clean up the data.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to be cleaned.
    Returns:
        pd.DataFrame: The cleaned DataFrame with specified columns converted to float.
    """
    df = df.drop(df[df['FG volume / year'].str.contains('-')].index)
    df['FG volume / year'] = df['FG volume / year'].astype(float)
    df['CIU curr / vol'] = df['CIU curr / vol'].astype(float)
    df['Flavor Spend'] = df['Flavor Spend'].astype(float)
    df['Total CIU curr / vol'] = df['Total CIU curr / vol'].astype(float)

    df['Multiple flavors'] = df['Multiple flavors'].fillna(0)
    df['Multiple flavors'] = df['Multiple flavors'].map(lambda x: 1 if x == 'x' else 0)
    df['Multiple flavors'] = df['Multiple flavors'].astype(int)

    return df
