import pandas as pd
import re


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

    # Используем поле description (если есть) вместо пустого snippet
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
import re


def infer_missing_fields_from_text(df: pd.DataFrame) -> pd.DataFrame:
    def extract_experience_level(text):
        if not isinstance(text, str):
            return None
        text = text.lower()

        if "без опыта" in text or "начинающий" in text or "стажёр" in text:
            return "Junior"
        if re.search(r"от\s?[1-2]\s?(год|лет)", text):
            return "Junior"
        if re.search(r"от\s?[3-4]\s?(год|лет)", text):
            return "Mid"
        if re.search(r"от\s?[5-9]\s?(год|лет)", text) or "большой опыт" in text:
            return "Senior"
        if re.search(r"(от\s?10|более 10)\s?(лет|лет опыта)", text):
            return "Executive"

        mid_markers = [
            "практический опыт", "участие в проектах", "самостоятельной подготовки",
            "опыт анализа", "опыт тестирования", "разработка", "внедрение", "оценка эффективности"
        ]
        for phrase in mid_markers:
            if phrase in text:
                return "Mid"

        junior_markers = [
            "базовые знания", "уверенный пользователь", "желание учиться", "начальные навыки"
        ]
        for phrase in junior_markers:
            if phrase in text:
                return "Junior"

        return None

    def extract_employment_type(text):
        if not isinstance(text, str):
            return None
        text = text.lower()

        full_time_markers = [
            "полная занятость", "полный рабочий день", "full-time", "работа на полный день"
        ]
        part_time_markers = [
            "частичная занятость", "неполный рабочий день", "part-time", "по совместительству"
        ]
        freelance_markers = [
            "фриланс", "удаленная работа", "удалённая работа", "работа удалённо", "remote work"
        ]
        contract_markers = [
            "контракт", "договор подряда", "временный проект", "по контракту", "project-based"
        ]

        for phrase in full_time_markers:
            if phrase in text:
                return "Full-time"
        for phrase in part_time_markers:
            if phrase in text:
                return "Part-time"
        for phrase in freelance_markers:
            if phrase in text:
                return "Freelance"
        for phrase in contract_markers:
            if phrase in text:
                return "Contract"

        return None

    hh_mask = df["source"] == "hh_api"

    full_text = pd.Series([
        f"{req or ''} {resp or ''}".lower()
        for req, resp in zip(df.get("requirement", ""), df.get("responsibility", ""))
    ], index=df.index)

    experience_extracted = full_text[hh_mask].apply(extract_experience_level)
    employment_extracted = full_text[hh_mask].apply(extract_employment_type)

    df.loc[hh_mask, "experience_level"] = experience_extracted.combine_first(df.loc[hh_mask, "experience_level"])
    df.loc[hh_mask, "employment_type"] = employment_extracted.combine_first(df.loc[hh_mask, "employment_type"])

    print(" NLP заполнение по тексту: найдено уровней опыта —", experience_extracted.notna().sum())
    print(" NLP заполнение по типу занятости —", employment_extracted.notna().sum())

    return df
