import pandas as pd
import mysql.connector


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="labor_market"
)
cursor = conn.cursor()


csv_path =  r"C:\Users\yfcnz\PycharmProjects\labor-market-analysis-project3\src\data\processed\combined_dataset_KT_format.csv"
df = pd.read_csv(csv_path)


cursor.execute("TRUNCATE TABLE combined_dataset_v2")


insert_query = """
INSERT INTO combined_dataset_v2 (
    work_year, job_title, experience_level, employment_type,
    salary, salary_currency, salary_in_usd,
    employee_residence, remote_ratio, company_location,
    company_size, source, employer, requirement, skills
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

for _, row in df.iterrows():
    cursor.execute(insert_query, tuple(row.fillna("").values))

conn.commit()
cursor.close()
conn.close()

print("Данные обновлены в базе данных.")
