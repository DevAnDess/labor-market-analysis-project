import pandas as pd
from sqlalchemy import create_engine

host = "sql7.freesqldatabase.com"
user = "sql7785672"
password = "DaVFNATBHJ"
database = "sql7785672"

engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

query = "SELECT * FROM combined_dataset_KT_format"

df = pd.read_sql(query, engine)

print("Подключение успешно.")
print(f"Получено строк: {len(df)}")
print(df.head())
