import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import streamlit as st

def hh_salary_hierarchical(selected_filters1, selected_filters2, salary_range):
    host = "sql10.freesqldatabase.com"
    user = "sql10783708"
    password = "7Izwc6qDZN"
    database = "sql10783708"

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    query = "SELECT * FROM combined_dataset_KT_format"

    df = pd.read_sql(query, engine)

    df = df[df['source'] == 'hh_api']
    df = df[
        df['company_size'].isin(selected_filters2) &
        df['experience_level'].isin(selected_filters1) &
        df['salary_in_usd'].between(salary_range[0], salary_range[1])
        ]

    df['salary_in_usd'] = df['salary_in_usd'] * 12

    desired_order = ['Unknown', 'Junior', 'Mid', 'Senior', 'Executive']
    df['experience_level'] = pd.Categorical(df['experience_level'], categories=desired_order, ordered=True)
    df['experience_level'] = df['experience_level'].cat.codes

    desired_order = ['S', 'M', 'L']
    df['company_size'] = pd.Categorical(df['company_size'], categories=desired_order, ordered=True)
    df['company_size'] = df['company_size'].cat.codes

    selected_df = df[['company_size', 'experience_level', 'salary_in_usd']].copy()
    x = selected_df[['company_size', 'experience_level', 'salary_in_usd']]

    if x.shape[0] < 10:
        st.warning(
            "No data matching the selected filters. Please adjust your filters. You need at least 10 data samples")
        return

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    linkage_matrix = linkage(x_scaled, method='ward')

    labels = fcluster(linkage_matrix, t=8, criterion='distance')
    selected_df['cluster'] = labels

    fig = px.scatter_3d(
        selected_df,
        x="company_size",
        y="experience_level",
        z="salary_in_usd",
        color="cluster",
        title="Company Size vs Experience Level vs Salary (3D)",
        labels={"experience_level": "Unknown, Junior, Mid, Senior, Executive", "salary_in_usd": "Salary in USD",
                "company_size": "Small, Medium, Large"},
    )
    st.plotly_chart(fig)

    fig = px.scatter(
        selected_df,
        x="company_size",
        y="salary_in_usd",
        color="cluster",
        title="Company Size vs Salary",
        labels={"salary_in_usd": "Salary in USD", "company_size": "Small, Medium, Large"},
    )
    st.plotly_chart(fig)

    fig = px.scatter(
        selected_df,
        x="experience_level",
        y="salary_in_usd",
        color="cluster",
        title="Experience Level vs Salary",
        labels={"salary_in_usd": "Salary in USD", "experience_level": "Unknown, Junior, Mid, Senior, Executive"},
    )
    st.plotly_chart(fig)