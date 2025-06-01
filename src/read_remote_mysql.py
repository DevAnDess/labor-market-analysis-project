import pandas as pd
from sqlalchemy import create_engine

user = "sql7782452"
password = "6HC3yNXWYM"
host = "sql7.freesqldatabase.com"
database = "sql7782452"

engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

query = "SELECT * FROM combined_dataset_KT_format"

df = pd.read_sql(query, engine)

print("Подключение успешно.")
print(f"Получено строк: {len(df)}")
print(df.head())
