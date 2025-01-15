import pandas as pd


def process_data(raw_data):
    # Преобразование JSON в DataFrame
    df = pd.DataFrame(raw_data)

    # Удаление ненужных колонок
    columns_to_keep = ["name", "salary", "area", "employer"]
    df = df[columns_to_keep]

    # Обработка пропущенных данных
    df["salary"] = df["salary"].fillna("Не указана")

    # Преобразование словарей в строковое представление
    for column in df.columns:
        df[column] = df[column].apply(lambda x: str(x) if isinstance(x, dict) else x)

    # Удаление дубликатов
    df = df.drop_duplicates()
    print(df.head())  # Покажет первые 5 строк данных
    print(df.dtypes)  # Покажет типы данных в каждом столбце

    return df
