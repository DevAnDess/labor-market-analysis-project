

def process_data(raw_data):
    df = pd.DataFrame(raw_data)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", None)

    columns_to_keep = [
        "name", "salary", "area", "employer", "schedule", "experience",
        "snippet", "published_at", "company_size"
    ]
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

    if "description" in df.columns:
        df["requirement"] = df["description"].fillna("Не указано")
    else:
        df["requirement"] = df["snippet"].apply(
            lambda x: x.get("requirement", "Не указано") if isinstance(x, dict) else "Не указано")

    df["responsibility"] = df["snippet"].apply(
        lambda x: x.get("responsibility", "Не указано") if isinstance(x, dict) else "Не указано")
    df = df.drop(columns=["snippet"], errors="ignore")

    df = df.dropna(subset=["name", "salary"]).drop_duplicates()

    return df


import pandas as pd

def infer_missing_fields_from_text(df: pd.DataFrame) -> pd.DataFrame:

    import re

    def extract_experience_level(text, title=""):
        if not isinstance(text, str):
            text = ""
        if not isinstance(title, str):
            title = ""

        text = text.lower()
        title = title.lower()


        if "начальный уровень" in title or "junior" in title:
            return "Junior"
        if "middle" in title or "средний уровень" in title:
            return "Mid"
        if "senior" in title or "ведущий" in title:
            return "Senior"
        if "lead" in title or "head" in title or "руководитель" in title:
            return "Executive"


        if re.search(r"от\s?10\s?лет", text) or "более 10 лет" in text:
            return "Executive"
        if re.search(r"от\s?[5-9]\s?(лет|года)", text) or "большой опыт" in text:
            return "Senior"
        if re.search(r"от\s?[3-4]\s?(лет|года)", text) or "самостоятельная работа" in text:
            return "Mid"
        if re.search(r"от\s?[1-2]\s?(лет|года)", text) or "без опыта" in text or "начинающий" in text:
            return "Junior"


        mid_markers = ["практический опыт", "участие в проектах", "внедрение решений", "ответственность за результат"]
        senior_markers = ["архитектура", "ведение команды", "руководство проектами"]
        junior_markers = ["базовые знания", "готовы обучать", "обучаемость", "начальные навыки"]

        if any(p in text for p in senior_markers):
            return "Senior"
        if any(p in text for p in mid_markers):
            return "Mid"
        if any(p in text for p in junior_markers):
            return "Junior"

        return None

    def extract_employment_type(text):
        if not isinstance(text, str):
            return None
        text = text.lower()

        if any(kw in text for kw in ["полная занятость", "полный рабочий день", "full-time"]):
            return "Full-time"
        if any(kw in text for kw in ["частичная занятость", "неполный рабочий день", "part-time"]):
            return "Part-time"
        if any(kw in text for kw in ["фриланс", "удалённая работа", "remote"]):
            return "Freelance"
        if any(kw in text for kw in ["контракт", "временный проект"]):
            return "Contract"
        return None

    hh_mask = df["source"] == "hh_api"


    if "requirement" not in df.columns:
        df["requirement"] = ""
    if "responsibility" not in df.columns:
        df["responsibility"] = ""

    full_text = df["requirement"].fillna("").astype(str) + " " + df["responsibility"].fillna("").astype(str)

    experience_extracted = df.loc[hh_mask].apply(
        lambda row: extract_experience_level(row.get("requirement", ""), row.get("job_title", "")), axis=1
    )

    employment_extracted = full_text[hh_mask].apply(extract_employment_type)

    df.loc[hh_mask, "experience_level"] = experience_extracted.combine_first(df.loc[hh_mask, "experience_level"])
    df.loc[hh_mask, "employment_type"] = employment_extracted.combine_first(df.loc[hh_mask, "employment_type"])

    print("NLP: найдено уровней опыта —", experience_extracted.notna().sum())
    print("NLP: найдено типов занятости —", employment_extracted.notna().sum())

    return df


