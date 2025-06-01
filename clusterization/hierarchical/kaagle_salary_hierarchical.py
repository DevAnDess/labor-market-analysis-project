import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

def kaagle_salary_hierarchical():
    user = "sql7782452"
    password = "6HC3yNXWYM"
    host = "sql7.freesqldatabase.com"
    database = "sql7782452"

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    query = "SELECT * FROM combined_dataset_KT_format"

    df = pd.read_sql(query, engine)


    df = df[(df['source'] == 'kaggle') & (df['experience_level'] != 'Unknown')]

    desired_order = ['Junior', 'Mid', 'Senior', 'Executive']
    df['experience_level'] = pd.Categorical(df['experience_level'], categories=desired_order, ordered=True)
    df['experience_level'] = df['experience_level'].cat.codes

    selected_df = df[['work_year', 'experience_level', 'salary_in_usd']].copy()
    x = selected_df[['work_year', 'experience_level', 'salary_in_usd']]


    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    linkage_matrix = linkage(x_scaled, method='ward')

    labels = fcluster(linkage_matrix, t=8, criterion='distance')
    selected_df['cluster'] = labels

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        selected_df['work_year'],
        selected_df['experience_level'],
        selected_df['salary_in_usd'],
        c=labels,
        cmap='viridis',
        s=50
    )
    ax.set_title("Work Year vs Experience Level vs Salary")
    ax.set_xlabel("Work Year")
    ax.set_ylabel("Junior, Mid, Senior, Executive")
    ax.set_zlabel("Salary in USD")
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.scatter(
        selected_df['work_year'],
        selected_df['salary_in_usd'],
        c=labels,
        cmap='viridis',
        s=50
    )
    plt.title("Work Year vs Salary")
    plt.xlabel("Work Year")
    plt.ylabel("Salary in USD")
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.scatter(
        selected_df['experience_level'],
        selected_df['salary_in_usd'],
        c=labels,
        cmap='viridis',
        s=50
    )
    plt.title("Experience Level vs Salary")
    plt.xlabel("Junior, Mid, Senior, Executive")
    plt.ylabel("Salary in USD")
    plt.grid(True)
    plt.show()
