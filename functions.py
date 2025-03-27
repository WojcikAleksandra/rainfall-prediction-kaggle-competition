import itertools
import numpy as np
import pandas as pd

def transform_circular(data, column):
    new_data = data.copy()
    new_data[column + '_sin'] = np.sin(2 * np.pi * new_data[column] / max(new_data[column]))
    new_data[column + '_cos'] = np.cos(2 * np.pi * new_data[column] / max(new_data[column]))
    return new_data

def add_combinations(df, exclude_columns=[]):
    selected_columns = [col for col in df.columns if col not in exclude_columns]

    new_columns = {}

    for n in range(2, 3):
        for cols in itertools.combinations(selected_columns, n):
            col_combination = '_'.join(cols)

            # Addition
            new_columns[f'{col_combination}_plus'] = df[list(cols)].sum(axis=1)

            # Subtraction
            if n == 2:
                new_columns[f'{cols[0]}_minus_{cols[1]}'] = df[cols[0]] - df[cols[1]]
                new_columns[f'{cols[1]}_minus_{cols[0]}'] = df[cols[1]] - df[cols[0]]
            else:
                for pair in itertools.combinations(cols, 2):
                    new_columns[f'{pair[0]}_minus_{pair[1]}'] = df[pair[0]] - df[pair[1]]
                    new_columns[f'{pair[1]}_minus_{pair[0]}'] = df[pair[1]] - df[pair[0]]

            # Multiplication
            new_columns[f'{col_combination}_times'] = df[list(cols)].prod(axis=1)

            # Division
            if n == 2:
                new_columns[f'{cols[0]}_div_{cols[1]}'] = df[cols[0]] / df[cols[1]]
                new_columns[f'{cols[1]}_div_{cols[0]}'] = df[cols[1]] / df[cols[0]]
            else:
                for pair in itertools.combinations(cols, 2):
                    new_columns[f'{pair[0]}_div_{pair[1]}'] = df[pair[0]] / df[pair[1]]
                    new_columns[f'{pair[1]}_div_{pair[0]}'] = df[pair[1]] / df[pair[0]]

    df_new = pd.DataFrame(new_columns, index=df.index)

    df_comb = pd.concat([df, df_new], axis=1)
    df_comb.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_comb.fillna(0, inplace=True)
    return df_comb

def add_prev_days(df):
    """
    This function creates `df_prev` dataframe from `df`, adding weather features from previous days.
    It adds 3 types of new columns:
    weather features from the previous day,
    the average values of features from the previous two days,
    and the average values from the previous three days.

    Parameters:
    df (pd.DataFrame): Input table with weather features (without 'id' column).

    Returns:
    df_prev (pd.DataFrame): A new table containing the original data and features from the previous days.
    """

    df_prev = df.copy()

    columns_to_drop = [col for col in df.columns if 'day' in col]

    df_shifted1 = df_prev.shift(1)
    df_shifted1.drop(columns_to_drop, axis=1, inplace=True)

    df_prev = pd.concat([df_prev, df_shifted1.add_prefix("prev1_")], axis=1)

    columns_to_drop = [col for col in df_shifted1.columns if any(keyword in col for keyword in ['winddirection', 'rainfall'])]

    df_shifted1.drop(columns_to_drop, axis=1, inplace=True)

    columns_to_drop = [col for col in df.columns if any(keyword in col for keyword in ['day', 'winddirection', 'rainfall'])]

    df_shifted2 = df.shift(2)
    df_shifted2.drop(columns_to_drop, axis=1, inplace=True)

    df_shifted3 = df.shift(3)
    df_shifted3.drop(columns_to_drop, axis=1, inplace=True)

    df_avg2 = pd.concat([df_shifted1, df_shifted2]).groupby(level=0).mean()
    df_avg2[(df_shifted1.isna()) | (df_shifted2.isna())] = np.nan
    df_avg2 = df_avg2.add_prefix("avg2_")
    df_avg2 = df_avg2.round(1)

    df_avg3 = pd.concat([df_shifted1, df_shifted2, df_shifted3]).groupby(level=0).mean()
    df_avg3[(df_shifted1.isna()) | (df_shifted2.isna()) | (df_shifted3.isna())] = np.nan
    df_avg3 = df_avg3.add_prefix("avg3_")
    df_avg3 = df_avg3.round(1)

    df_prev = pd.concat([df_prev, df_avg2, df_avg3], axis=1)
    # replace inf with nan and nan with 0
    df_prev.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prev.fillna(0, inplace=True)
    #df_prev = df_prev.dropna().reset_index(drop=True)

    return df_prev
#%%
