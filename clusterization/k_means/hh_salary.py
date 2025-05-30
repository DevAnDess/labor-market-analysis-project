import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def hh_salary():
    df = pd.read_csv("../src/data/processed/combined_dataset_KT_format.csv")
    df = df[(df['source'] == 'hh_api') & (df['experience_level'] != 'Unknown') & (df['salary_in_usd'] != 'Unknown') & (df['company_size'] != 'Unknown')]

    df['salary_in_usd'] = df['salary_in_usd'] * 12

    df['experience_level'] = df['experience_level'].astype('category').cat.codes

    df['company_size'] = df['company_size'].astype('category').cat.codes

    selected_df = df[['company_size', 'experience_level', 'salary_in_usd']].copy()
    x = selected_df[['company_size', 'experience_level', 'salary_in_usd']]


    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(x)
    labels = kmeans.labels_

    selected_df['cluster'] = labels

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        x['company_size'],
        x['experience_level'],
        x['salary_in_usd'],
        c=labels,
        cmap='viridis',
        s=50
    )

    ax.set_title("Company size vs Experience Level vs Salary")
    ax.set_xlabel("Small, Medium, Large")
    ax.set_ylabel("Junior, Mid, Senior, Executive")
    ax.set_zlabel("Salary in USD")
    plt.show()

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(x)
    labels = kmeans.labels_
    selected_df['cluster'] = labels

    plt.figure(figsize=(8, 6))
    plt.scatter(
        selected_df['company_size'],
        selected_df['salary_in_usd'],
        c=labels,
        cmap='viridis',
        s=50
    )
    plt.title("Company size vs Salary")
    plt.xlabel("Small, Medium, Large")
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