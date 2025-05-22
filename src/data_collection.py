import os
import requests
import pandas as pd

from src.data_processing import process_data
from src.data_analysis import extract_min_salary

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def fetch_all_vacancies(query: str = "аналитик данных", area: int = 1, max_pages: int = 10) -> list[dict]:
    url = "https://api.hh.ru/vacancies"
    all_vacancies = []
    page = 0

    while page < max_pages:
        params = {
            "text": query,
            "area": area,
            "per_page": 100,
            "page": page
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"HH API returned {resp.status_code} on page {page}")
            break

        payload = resp.json()
        items = payload.get("items", [])
        all_vacancies.extend(items)
        print(f"Собрано {len(items)} вакансий с страницы {page + 1}")

        if not items or page >= payload.get("pages", 1) - 1:
            break
        page += 1

    print(f"Всего собрано {len(all_vacancies)} вакансий из HH API")
    return all_vacancies


def fetch_kaggle_dataset_local(csv_filename: str = "data-science-job-salaries.csv") -> pd.DataFrame:
    SRC_DIR = os.path.dirname(__file__)
    kaggle_dir = os.path.join(SRC_DIR, "data", "raw", "kaggle")
    csv_path = os.path.join(kaggle_dir, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV-файл не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} строк из {csv_path}")
    return df


def _normalize_kaggle_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        rec = {
            "name": row["job_title"],
            "salary": {
                "from": row["salary_in_usd"],
                "to": row["salary_in_usd"],
                "currency": "USD"
            },
            "numeric_salary": row["salary_in_usd"],
            "min_salary": row["salary_in_usd"],
            "area": {"name": row["employee_residence"]},
            "employer": {"name": None},
            "schedule": {"name": row["employment_type"]},
            "experience": {"name": row["experience_level"]},
            "company_size": row["company_size"],
            "snippet": {
                "requirement": f"remote_ratio={row['remote_ratio']}",
                "responsibility": f"company_location={row['company_location']}"
            },
            "published_at": row["work_year"]
        }
        records.append(rec)
    return records


def fetch_and_combine(query: str = "аналитик данных",
                      area: int = 1,
                      csv_filename: str = "data-science-job-salaries.csv") -> pd.DataFrame:
    import os
    import pandas as pd
    from src.data_processing import process_data
    from src.data_analysis import extract_min_salary

    # Kaggle
    kaggle_df = fetch_kaggle_dataset_local(csv_filename)
    kaggle_raw = _normalize_kaggle_records(kaggle_df)
    df_kaggle = process_data(kaggle_raw)
    df_kaggle["source"] = "kaggle"
    df_kaggle["min_salary"] = df_kaggle["salary"].apply(extract_min_salary)

    # HH API
    hh_raw = fetch_all_vacancies(query=query, area=area)
    df_hh = process_data(hh_raw)
    df_hh["source"] = "hh_api"
    df_hh["min_salary"] = df_hh["salary"].apply(extract_min_salary)

    # Объединение
    expected_cols = [
        "name", "salary", "numeric_salary", "min_salary", "area",
        "employer", "schedule", "experience", "company_size",
        "requirement", "responsibility", "published_at", "source"
    ]
    for df in (df_kaggle, df_hh):
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

    combined = pd.concat([df_kaggle[expected_cols], df_hh[expected_cols]], ignore_index=True)
    combined.drop_duplicates(subset=["name", "employer", "published_at", "area"], keep="first", inplace=True)


    combined.rename(columns={
        "name": "job_title",
        "area": "employee_residence",
        "experience": "experience_level",
        "schedule": "employment_type",
        "numeric_salary": "salary_in_usd"
    }, inplace=True)


    combined["salary_currency"] = combined["salary"].apply(
        lambda x: x.split()[-1] if isinstance(x, str) and len(x.split()) > 1 else "USD"
    )
    combined["salary_in_usd"] = pd.to_numeric(combined["salary_in_usd"], errors="coerce")
    # Конвертация в доллары
    usd_to_rub = 90
    combined.loc[combined["salary_currency"] == "RUR", "salary_in_usd"] = \
        combined.loc[combined["salary_currency"] == "RUR", "salary_in_usd"] // usd_to_rub

    experience_map = {
        "EN": "Junior", "MI": "Mid", "SE": "Senior", "EX": "Executive",
        "Junior": "Junior", "Mid": "Mid", "Senior": "Senior", "Executive": "Executive"
    }
    combined["experience_level"] = combined["experience_level"].map(experience_map).fillna("Unknown")


    employment_map = {
        "FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance",
        "Full-time": "Full-time", "Part-time": "Part-time", "Contract": "Contract", "Freelance": "Freelance"
    }
    combined["employment_type"] = combined["employment_type"].map(employment_map).fillna("Unknown")


    combined["remote_ratio"] = combined["requirement"].str.extract(r"remote_ratio=(\d+)").astype("Int64")
    combined["company_location"] = combined["responsibility"].str.extract(r"company_location=([A-Z]{2})")


    combined.loc[combined["source"] == "hh_api", "company_location"] = "RU"


    combined["company_size"] = combined["company_size"].fillna("Unknown")


    required_columns = [
        "published_at", "job_title", "experience_level", "employment_type",
        "salary", "salary_currency", "salary_in_usd",
        "employee_residence", "remote_ratio", "company_location", "company_size",
        "source", "employer"
    ]
    combined = combined[required_columns]
    country_map = {
        "RU": "Russia", "US": "United States", "GB": "United Kingdom", "DE": "Germany", "FR": "France",
        "CA": "Canada", "IN": "India", "JP": "Japan", "CN": "China", "PL": "Poland",
        "HN": "Honduras", "ES": "Spain", "NL": "Netherlands", "IT": "Italy",
        "UA": "Ukraine", "CH": "Switzerland", "BE": "Belgium", "PT": "Portugal",
        "BR": "Brazil", "MX": "Mexico", "AU": "Australia", "IE": "Ireland",
        "LT": "Lithuania", "CZ": "Czech Republic", "AT": "Austria", "SE": "Sweden"
    }


    combined["employee_residence"] = combined["employee_residence"].map(country_map).fillna(
        combined["employee_residence"])
    combined["company_location"] = combined["company_location"].map(country_map).fillna(combined["company_location"])

    combined.rename(columns={"published_at": "work_year"}, inplace=True)

    combined["work_year"] = combined["work_year"].astype(str).str[:4]

    proc_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    out_csv = os.path.join(proc_dir, "combined_dataset_KT_format.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Финальный датасет сохранён в: {out_csv}")

    return combined
