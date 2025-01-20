import pandas as pd


def process_data(raw_data):
    df = pd.DataFrame(raw_data)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", None)

    columns_to_keep = ["name", "salary", "area", "employer", "schedule", "experience", "snippet", "published_at"]
    df = df[[col for col in columns_to_keep if col in df.columns]]

    df["area"] = df["area"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)
    df["schedule"] = df["schedule"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)
    df["experience"] = df["experience"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)

    def extract_salary(salary):
        if isinstance(salary, dict):
            min_salary = salary.get("from") or salary.get("min")
            max_salary = salary.get("to") or salary.get("max")
            currency = salary.get("currency", "RUR")

            if min_salary is None and max_salary is None:
                return "Не указано"
            elif min_salary is None:
                return f"{max_salary} {currency}"
            else:
                return f"{min_salary} - {max_salary} {currency}"
        return "Не указано"

    df["salary"] = df["salary"].apply(extract_salary)

    df["numeric_salary"] = df["salary"].apply(
        lambda x: int(x.split(" ")[0]) if isinstance(x, str) and x.split(" ")[0].isdigit() else 0)

    df["employer"] = df["employer"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)

    df["requirement"] = df["snippet"].apply(
        lambda x: x.get("requirement", "Не указано") if isinstance(x, dict) else "Не указано")
    df["responsibility"] = df["snippet"].apply(
        lambda x: x.get("responsibility", "Не указано") if isinstance(x, dict) else "Не указано")
    df = df.drop(columns=["snippet"], errors="ignore")

    df = df.dropna(subset=["name", "salary"]).drop_duplicates()

    #    print("\nОбработанные данные (первые 10 строк):")
    #    print(df.head(10))

    return df
