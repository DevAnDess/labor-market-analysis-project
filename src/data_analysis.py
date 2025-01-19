import pandas as pd


def analyze_data(data):
    data["min_salary"] = data["salary"].apply(
        lambda x: int(x.split(" - ")[0]) if " - " in x and x.split(" - ")[0].isdigit() else None)

    grouped_by_area = data.groupby("area").size().reset_index(name="Количество вакансий")

    avg_salary_by_area = data.groupby("area")["numeric_salary"].mean().reset_index(name="Средняя зарплата")

    top_companies = data["employer"].value_counts().head(10).reset_index()
    top_companies.columns = ["Компания", "Количество вакансий"]

    print("\nГруппировка по регионам:")
    print(grouped_by_area)

    print("\nСредняя зарплата по регионам:")
    print(avg_salary_by_area)

    print("\nТОП-10 компаний по количеству вакансий:")
    print(top_companies)


    return {
        "grouped_by_area": grouped_by_area,
        "avg_salary_by_area": avg_salary_by_area,
        "top_companies": top_companies
    }
