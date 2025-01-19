import pandas as pd


def process_data(raw_data):
    df = pd.DataFrame(raw_data)
    pd.set_option("display.max_rows", 10)
    columns_to_keep = ["name", "salary", "area", "employer"]
    df = df[columns_to_keep]
    df["area"] = df["area"].apply(lambda x: x.get("name", "Не указано") if isinstance(x, dict) else x)

    df["salary"] = df["salary"].fillna("Не указана")

    def format_salary(salary):
        if isinstance(salary, dict):
            min_salary = salary.get("min", "Не указано")
            max_salary = salary.get("max", "Не указано")
            return f"{min_salary} - {max_salary}"
        return salary

    df["salary"] = df["salary"].apply(format_salary)

    def format_employer(employer):
        if isinstance(employer, dict):
            return employer.get("name", "Не указано")
        return employer

    def extract_salary(salary):
        if isinstance(salary, str) and " - " in salary:
            min_salary = salary.split(" - ")[0]
            return int(min_salary) if min_salary.isdigit() else None
        return None

    df["numeric_salary"] = df["salary"].apply(extract_salary)

    df["employer"] = df["employer"].apply(format_employer)

    for column in df.columns:
        df[column] = df[column].apply(lambda x: str(x) if isinstance(x, dict) else x)

    print(df.head())
    df = df.drop_duplicates()

    return df
