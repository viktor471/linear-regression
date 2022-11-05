import pandas as pd
from scipy.io import arff

def get_dataframe_from_arff(path: str) -> pd.DataFrame:

    data = arff.loadarff("dataset/dataset_191_wine.arff")
    return pd.DataFrame(data[0])
