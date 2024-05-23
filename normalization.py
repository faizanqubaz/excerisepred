from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize_data(df):
    ct = ColumnTransformer(transformers=[
        ('n',StandardScaler(),['deneyim','maas'])
    ])

    ct_trans = ct.fit_transform(df)

    ct_columns = ct.get_feature_names_out()


    return pd.DataFrame(ct_trans,columns=ct_columns)
