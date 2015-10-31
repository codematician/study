"""Some utilities for machine learning algorithms"""

from math import log2

def majority_val(df, attr):
    """Gives the majority value of column attr in dataframe df"""
    if len(df) == 0:
        return None
    mode = df[attr].mode()
    return mode[0] if len(mode) > 0 else df[attr].iloc[0]


def series_info(series):
    """Returns the bits of information required to represent a series"""
    return sum([-p * log2(p) for p in series.groupby(lambda x: series[x]).count() / len(series)])
