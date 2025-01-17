import pandas as pd


def process_data(raw_data):
    df = pd.DataFrame(raw_data)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", None)

    columns_to_keep = ["name", "salary", "area", "employer", "snippet", "published_at", "schedule", "experience"]
    df = df[[col for col in columns_to_keep if col in df.columns]]

    df["salary"] = df["salary"].fillna("Не указано")
    df["schedule"] = df["schedule"].fillna("Не указано")
    df["experience"] = df["experience"].fillna("Не указано")
    df["published_at"] = df["published_at"].fillna("Не указано")
    df["snippet"] = df["snippet"].fillna("Не указано")

    def format_salary(salary):
        if isinstance(salary, dict):
            min_salary = salary.get("from") or salary.get("min") or "Не указано"
            max_salary = salary.get("to") or salary.get("max") or "Не указано"
            currency = salary.get("currency", "RUR")
            if min_salary == "Не указано" and max_salary == "Не указано":
                return "Не указано"
            return f"{min_salary} - {max_salary} {currency}"
        return "Не указано"

    df["salary"] = df["salary"].apply(format_salary)

    def format_employer(employer):
        if isinstance(employer, dict):
            return employer.get("name", "Не указано")
        return employer

    df["employer"] = df["employer"].apply(format_employer)

    df["area"] = df["area"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)
    df["schedule"] = df["schedule"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)
    df["experience"] = df["experience"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)

    df["requirement"] = df["snippet"].apply(lambda x: x.get("requirement", "Не указано") if isinstance(x, dict) else "Не указано")
    df["responsibility"] = df["snippet"].apply(lambda x: x.get("responsibility", "Не указано") if isinstance(x, dict) else "Не указано")
    df = df.drop(columns=["snippet"])

    print(df.head(10))  # Проверка зарплат

    df = df.drop_duplicates()


    return df
