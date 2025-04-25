"""Module containing helper functions for the data understanding section"""

import numpy as np
import pandas as pd


def describe_numerical(data, numerical_cols):
    """Return a table containing summary statistics for each numerical
    feature in data"""
    numerical_eda = data.loc[:, numerical_cols].describe().T
    numerical_eda.loc[:, '% missing'] = (
        data.loc[:, numerical_cols].isnull().sum() / len(data) * 100
    )

    return numerical_eda.loc[:, ['count', '% missing', 'min', '25%', 'mean',
                                 '50%', '75%', 'max', 'std']]


def describe_categorical(data, categorical_cols):
    """Return a table containing summary statistics for each categorical
    feature in data"""
    categorical_eda = data.loc[:, categorical_cols].agg(['count', 'nunique']).T
    categorical_eda.loc[:, '% missing'] = (
        data.loc[:, categorical_cols].isnull().sum() / len(data) * 100
    )

    # Compute the mode, mode frequency and % using value counts
    modes = {}
    for col in categorical_cols:
        cur_modes = {}
        value_counts = data.loc[:, col].value_counts().nlargest(2)
        cur_modes['mode'] = value_counts.index[0]
        cur_modes['mode freq'] = int(value_counts.iloc[0])
        cur_modes['mode %'] = value_counts.iloc[0] / len(data) * 100
        cur_modes['2nd mode'] = value_counts.index[1]
        cur_modes['2nd mode freq'] = value_counts.iloc[1]
        cur_modes['2nd mode %'] = value_counts.iloc[0] / len(data) * 100
        modes[col] = cur_modes
    modes = pd.DataFrame(modes).T.astype(
        {'mode freq': int, '2nd mode freq': int}
    )

    # Join modes to categorical EDA
    categorical_eda = categorical_eda.join(modes)

    return categorical_eda.loc[:, ['count', '% missing', 'nunique', 'mode',
                                   'mode freq', 'mode %', '2nd mode',
                                   '2nd mode freq', '2nd mode %']]
