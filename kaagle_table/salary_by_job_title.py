import pandas as pd
import matplotlib.pyplot as plt

def average_sal():
    df = pd.read_csv("kaagle_table.csv")

    average_salaries = df.groupby('job_title')['salary_in_usd'].mean().reset_index()
    average_salaries.columns = ['job_title', 'average_salary_in_usd_by_job_title']

    top_ten_jobs = average_salaries.nlargest(10, 'average_salary_in_usd_by_job_title').sort_values(by='average_salary_in_usd_by_job_title', ascending=False)

    print(average_salaries)

    plt.figure(figsize=(10, 5))
    plt.bar(top_ten_jobs['job_title'], top_ten_jobs['average_salary_in_usd_by_job_title'], color='purple')
    plt.xlabel('job_title')
    plt.ylabel('average_salary_in_usd_by_job_title')
    plt.title('Top 10 Average Data Science Salaries by Job Title')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

average_sal()