import os
import pandas as pd

from src.data_processing import process_data, infer_missing_fields_from_text
from src.data_analysis import extract_min_salary, extract_skills
import time
import requests
from tqdm import tqdm


def enrich_hh_descriptions(vacancies: list[dict], sleep_sec: float = 0.2) -> list[dict]:
    enriched = []

    print(f"\n Обогащаем вакансии HH подробными описаниями ({len(vacancies)} вакансий)...")
    for vac in tqdm(vacancies, desc="Обработка вакансий", unit="вакансий"):
        vacancy_id = vac.get("id")
        if not vacancy_id:
            vac["description"] = ""
            enriched.append(vac)
            continue

        try:
            resp = requests.get(f"https://api.hh.ru/vacancies/{vacancy_id}")
            if resp.status_code == 200:
                vacancy_data = resp.json()
                vac["description"] = vacancy_data.get("description", "")
            else:
                vac["description"] = ""
        except Exception as e:
            print(f"[!] Ошибка при запросе ID {vacancy_id}: {e}")
            vac["description"] = ""

        enriched.append(vac)
        time.sleep(sleep_sec)  # защита от блокировки API

    print("описание добавлено к вакансиям.")
    return enriched


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def fetch_all_vacancies(query: str = "аналитик данных", area: int = 1, max_pages: int = 50) -> list[dict]:
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


def assign_company_size_from_lists(df: pd.DataFrame) -> pd.DataFrame:
    import os


    base_dir = os.path.dirname(__file__)
    ref_dir = os.path.join(base_dir, "data", "reference")
    large_path = os.path.join(ref_dir, "large_companies.txt")
    medium_path = os.path.join(ref_dir, "medium_companies.txt")


    def load_company_list(path):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return set(line.strip().lower() for line in f if line.strip())
        return set()

    large = load_company_list(large_path)
    medium = load_company_list(medium_path)

    def detect_size(name):
        if not isinstance(name, str) or name.strip() == "":
            return "Unknown"
        name = name.lower()
        if name in large:
            return "L"
        elif name in medium:
            return "M"
        else:
            return "S"


    mask = df["company_size"].isin(["Unknown", None, pd.NA])
    df.loc[mask, "company_size"] = df.loc[mask, "employer"].apply(detect_size)

    return df
def load_custom_dataset(filename="data_science_salaries.csv") -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "data", "raw", "kaggle", filename)
    df = pd.read_csv(path)

    remote_map = {
        "Remote": 100,
        "On-site": 0,
        "Hybrid": 50
    }
    df["remote_ratio"] = df["work_models"].map(remote_map)
    df.drop(columns=["work_models"], inplace=True)

    company_size_map = {"Medium": "M", "Large": "L", "Small": "S", "Unknown": "Unknown"}
    df["company_size"] = df["company_size"].map(company_size_map)


    exp_map = {
        "Junior-level": "Junior",
        "Mid-level": "Mid",
        "Senior-level": "Senior",
        "Executive-level": "Executive"
    }
    df["experience_level"] = df["experience_level"].map(exp_map).fillna("Unknown")

    df["source"] = "kaggle"
    df["employer"] = None
    df["requirement"] = None
    df["skills"] = None

    return df[[
        "job_title", "experience_level", "employment_type", "work_year",
        "employee_residence", "salary", "salary_currency", "salary_in_usd",
        "company_location", "company_size", "source", "employer", "requirement", "skills", "remote_ratio"
    ]]
def fetch_and_combine(
    query: str = "аналитик данных",
    area: int = 1,
    csv_filename: str = "data-science-job-salaries.csv"
) -> pd.DataFrame:

    kaggle_df = fetch_kaggle_dataset_local(csv_filename)
    kaggle_normalized = _normalize_kaggle_records(kaggle_df)
    df_kaggle = process_data(kaggle_normalized)
    df_kaggle["source"] = "kaggle"
    df_kaggle["min_salary"] = df_kaggle["salary"].apply(extract_min_salary)


    hh_all = fetch_all_vacancies(query=query, area=area)

    title_keywords = ["аналитик", "analyst", "data", "bi", "machine learning"]
    stopwords = ["директор", "manager", "менеджер", "qa", "тестировщик", "юрис", "маркетинг", "админ", "руководитель"]

    def is_relevant(vac):
        title = (vac.get("name") or "").lower()
        return any(kw in title for kw in title_keywords) and not any(sw in title for sw in stopwords)

    hh_filtered = [v for v in hh_all if is_relevant(v)]
    hh_enriched = enrich_hh_descriptions(hh_filtered)
    df_hh = process_data(hh_enriched)
    df_hh["source"] = "hh_api"
    df_hh["min_salary"] = df_hh["salary"].apply(extract_min_salary)


    expected_cols = [
        "name", "salary", "numeric_salary", "min_salary", "area",
        "employer", "schedule", "experience", "company_size",
        "requirement", "published_at", "source", "skills"
    ]
    for df in (df_kaggle, df_hh):
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

    combined = pd.concat([df_kaggle[expected_cols], df_hh[expected_cols]], ignore_index=True)


    combined.rename(columns={
        "name": "job_title",
        "area": "employee_residence",
        "experience": "experience_level",
        "schedule": "employment_type",
        "numeric_salary": "salary_in_usd",
        "published_at": "work_year"
    }, inplace=True)

    combined.drop_duplicates(
        subset=["job_title", "work_year", "employee_residence", "salary_in_usd"],
        keep="first", inplace=True
    )

    # Валюта
    combined["salary_currency"] = combined["salary"].apply(
        lambda x: x.split()[-1] if isinstance(x, str) and len(x.split()) > 1 else "USD"
    )
    combined["salary_in_usd"] = pd.to_numeric(combined["salary_in_usd"], errors="coerce")
    combined.loc[combined["salary_currency"] == "RUR", "salary_in_usd"] //= 90


    experience_map = {
        "EN": "Junior", "MI": "Mid", "SE": "Senior", "EX": "Executive",
        "Junior": "Junior", "Mid": "Mid", "Senior": "Senior", "Executive": "Executive"
    }
    employment_map = {
        "FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance",
        "Full-time": "Full-time", "Part-time": "Part-time", "Contract": "Contract", "Freelance": "Freelance"
    }
    combined["experience_level"] = combined["experience_level"].map(experience_map).fillna("Unknown")
    combined["employment_type"] = combined["employment_type"].map(employment_map).fillna("Full-time")


    combined["remote_ratio"] = combined["requirement"].str.extract(r"remote_ratio=(\d+)").astype("Int64")
    if "responsibility" in combined.columns:
        combined["company_location"] = combined["responsibility"].str.extract(r"company_location=([A-Z]{2})")
        combined.drop(columns=["responsibility"], inplace=True)
    else:
        combined["company_location"] = pd.NA

    combined.loc[combined["source"] == "hh_api", "company_location"] = "RU"
    combined["company_size"] = combined["company_size"].fillna("Unknown")
    if 'responsibility' in combined.columns:
        combined.drop(columns=['responsibility'], inplace=True)


    country_map = {
        "RU": "Russia", "US": "United States", "GB": "United Kingdom", "DE": "Germany", "FR": "France",
        "CA": "Canada", "IN": "India", "JP": "Japan", "CN": "China", "PL": "Poland", "HN": "Honduras",
        "ES": "Spain", "NL": "Netherlands", "IT": "Italy", "UA": "Ukraine", "CH": "Switzerland",
        "BE": "Belgium", "PT": "Portugal", "BR": "Brazil", "MX": "Mexico", "AU": "Australia",
        "IE": "Ireland", "LT": "Lithuania", "CZ": "Czech Republic", "AT": "Austria", "SE": "Sweden"
    }
    combined["employee_residence"] = combined["employee_residence"].map(country_map).fillna(combined["employee_residence"])
    combined["company_location"] = combined["company_location"].map(country_map).fillna(combined["company_location"])
    combined["work_year"] = combined["work_year"].astype(str).str[:4]


    final_cols = [
        "work_year", "job_title", "experience_level", "employment_type",
        "salary", "salary_currency", "salary_in_usd",
        "employee_residence", "remote_ratio", "company_location", "company_size",
        "source", "employer", "requirement", "skills"
    ]
    custom_df = load_custom_dataset("data_science_salaries.csv")
    combined = pd.concat([combined, custom_df], ignore_index=True)
    combined = combined[final_cols]
    combined["skills"] = combined["requirement"].fillna("").astype(str).apply(extract_skills)
    combined.drop(columns=["responsibility"], errors="ignore", inplace=True)

    combined = infer_missing_fields_from_text(combined)
    combined = assign_company_size_from_lists(combined)

    proc_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    out_csv = os.path.join(proc_dir, "combined_dataset_KT_format.csv")
    combined.to_csv(out_csv, index=False)

    return combined

