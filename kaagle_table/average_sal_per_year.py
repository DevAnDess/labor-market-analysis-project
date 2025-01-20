import pandas as pd
import matplotlib.pyplot as plt

def average_sal():
    df = pd.read_csv("kaagle_table.csv")

    average_salaries = df.groupby('work_year')['salary_in_usd'].mean().reset_index()
    average_salaries.columns = ['year', 'average_salary_in_usd_by_year']

    print(average_salaries // 1)

    plt.figure(figsize=(10, 5))
    plt.bar(average_salaries['year'], average_salaries['average_salary_in_usd_by_year'], color='purple')
    plt.xlabel('year')
    plt.ylabel('average_salary_in_usd_by_year')
    plt.title('Average Data Science Salaries per Year')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

average_sal()