import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def hh_salary():
    df = pd.read_csv("../src/data/processed/combined_dataset_KT_format.csv")
    df = df[534:]
    print(df)