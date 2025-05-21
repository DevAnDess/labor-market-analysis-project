import pandas as pd


def extract_min_salary(salary):
    if isinstance(salary, str) and " - " in salary:
        min_salary = salary.split(" - ")[0]
        return int(min_salary) if min_salary.isdigit() else None
    return None


def analyze_data(data):
    data["min_salary"] = data["salary"].apply(extract_min_salary)
    grouped_by_area = data.groupby("employee_residence").size().reset_index(name="Количество вакансий")
    grouped_by_area.rename(columns={"employee_residence": "Регион"}, inplace=True)

    if "salary_in_usd" in data.columns:
        data["salary_in_usd"] = data["salary_in_usd"].fillna(0).infer_objects(copy=False)

        avg_salary_by_area = data.groupby("employee_residence")["salary_in_usd"].mean().reset_index(
            name="Средняя зарплата")
        avg_salary_by_area.rename(columns={"employee_residence": "Регион"}, inplace=True)
    else:
        avg_salary_by_area = pd.DataFrame(columns=["Регион", "Средняя зарплата"])

    top_companies = data["employer"].value_counts().head(10).reset_index()
    top_companies.columns = ["Компания", "Количество вакансий"]

    return {
        "grouped_by_area": grouped_by_area,
        "avg_salary_by_area": avg_salary_by_area,
        "top_companies": top_companies
    }
