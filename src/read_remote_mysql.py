import pandas as pd
from sqlalchemy import create_engine

host = "sql10.freesqldatabase.com"
user = "sql10783708"
password = "7Izwc6qDZN"
database = "sql10783708"

engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

query = "SELECT * FROM combined_dataset_KT_format"

df = pd.read_sql(query, engine)

print("Подключение успешно.")
print(f"Получено строк: {len(df)}")
print(df.head())
