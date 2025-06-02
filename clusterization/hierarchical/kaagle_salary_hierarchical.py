import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import streamlit as st

def kaagle_salary_hierarchical(selected_filters1, selected_filters2, salary_range):
    user = "sql7782452"
    password = "6HC3yNXWYM"
    host = "sql7.freesqldatabase.com"
    database = "sql7782452"

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    query = "SELECT * FROM combined_dataset_KT_format"

    df = pd.read_sql(query, engine)

    df = df[
        df['work_year'].isin(selected_filters1) &
        df['experience_level'].isin(selected_filters2) &
        df['salary_in_usd'].between(salary_range[0], salary_range[1])
        ]

    df = df[(df['source'] == 'kaggle') & (df['experience_level'] != 'Unknown')]

    desired_order = ['Junior', 'Mid', 'Senior', 'Executive']
    df['experience_level'] = pd.Categorical(df['experience_level'], categories=desired_order, ordered=True)
    df['experience_level'] = df['experience_level'].cat.codes

    selected_df = df[['work_year', 'experience_level', 'salary_in_usd']].copy()
    x = selected_df[['work_year', 'experience_level', 'salary_in_usd']]

    if x.shape[0] < 10:
        st.warning("No data matching the selected filters. Please adjust your filters. You need at least 10 data samples")
        return

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    linkage_matrix = linkage(x_scaled, method='ward')

    labels = fcluster(linkage_matrix, t=8, criterion='distance')
    selected_df['cluster'] = labels

    fig = px.scatter_3d(
        selected_df,
        x="work_year",
        y="experience_level",
        z="salary_in_usd",
        color="cluster",
        title="Work Year vs Experience Level vs Salary (3D)",
        labels={"experience_level": "Junior, Mid, Senior, Executive", "salary_in_usd": "Salary in USD", "work_year": "Work Year"},
    )
    st.plotly_chart(fig)

    fig = px.scatter(
        selected_df,
        x="work_year",
        y="salary_in_usd",
        color="cluster",
        title="Work Year vs Salary",
        labels={"salary_in_usd": "Salary in USD", "work_year": "Work Year"},
    )
    st.plotly_chart(fig)

    fig = px.scatter(
        selected_df,
        x="experience_level",
        y="salary_in_usd",
        color="cluster",
        title="Experience Level vs Salary",
        labels={"experience_level": "Junior, Mid, Senior, Executive", "salary_in_usd": "Salary in USD"},
    )
    st.plotly_chart(fig)
