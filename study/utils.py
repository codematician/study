

def majority_val(df, attr):
    """Gives the majority value of column attr in dataframe df"""
    if len(df) == 0:
        return None
    mode = df[attr].mode()
    return mode[0] if len(mode) > 0 else df[attr].iloc[0]
