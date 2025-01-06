import pandas as pd

def one_hot_encoder(df):
    columns = list(df.columns)
    cat_col = [cat for cat in columns if df[cat].dtype == "object" or df[cat].dtype == "category"]
    df = pd.get_dummies(df,columns=cat_col)
    new_cols = [c for c in df.columns if c not in columns]
    return df, new_cols