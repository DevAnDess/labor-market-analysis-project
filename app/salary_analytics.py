import streamlit as st
import pandas as pd

st.set_page_config(page_title="Software for Data Analytics Labor Market Analysis", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #181818;
            color: #cccccc;
        }
        .stApp {
            background-color: #181818;
        }
        h1 {
            color: #fff !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color: white;'>Software for Data Analytics Labor Market Analysis</h1>",
    unsafe_allow_html=True
)

df = pd.read_csv("D:/labor-market-analysis-project/src/data/processed/combined_dataset_KT_format.csv")

st.dataframe(df, use_container_width=True)