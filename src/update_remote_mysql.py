import pandas as pd
import mysql.connector
from mysql.connector import Error

csv_path = r"C:\Users\yfcnz\PycharmProjects\labor-market-analysis-project3\src\data\processed\combined_dataset_KT_format.csv"

df = pd.read_csv(csv_path)

try:
    conn = mysql.connector.connect(
        host="sql7.freesqldatabase.com",
        user="sql7782452",
        password="6HC3yNXWYM",
        database="sql7782452"
    )
    cursor = conn.cursor()
    print("Подключение успешно")

    try:
        cursor.execute("DELETE FROM combined_dataset_KT_format")
        print("Старые данные очищены.")
    except Error as e:
        print("Ошибка удаления данных:", e)

    insert_query = """
        INSERT INTO combined_dataset_KT_format (
            work_year, job_title, experience_level, employment_type,
            salary, salary_currency, salary_in_usd, employee_residence,
            remote_ratio, company_location, company_size, source,
            employer, requirement, skills
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for i, (_, row) in enumerate(df.iterrows(), 1):
        cursor.execute(insert_query, tuple(row.fillna("").values))
        if i % 1000 == 0:
            print(f"Загружено строк: {i}")

    conn.commit()
    print("Данные обновлены на сервере")

except Error as e:
    print("Ошибка подключения или выполнения запроса:", e)

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals() and conn.is_connected():
        conn.close()
