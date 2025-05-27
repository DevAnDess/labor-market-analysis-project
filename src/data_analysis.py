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
import spacy
nlp = spacy.load("ru_core_news_md")



tech_keywords = {"python", "sql", "tableau", "power bi", "excel", "looker", "airflow", "vba", "postgresql", "mongodb",
                 "spark", "hadoop", "r", "etl", "sas", "bi", "jira", "confluence", "docker", "git", "snowflake", "bash",
                 "linux"}


custom_stopwords = {"опыт", "знание", "умение", "работа", "понимание", "highlighttext", "данные", "анализ", "год",
                    "владение", "применение", "использование", "область"}

def extract_skills(text):
    if not isinstance(text, str):
        return []

    doc = nlp(text.lower())
    skills = set()

    for token in doc:
        if token.is_stop or token.is_punct or token.lemma_ in custom_stopwords:
            continue
        if token.lemma_.isalpha() and token.pos_ == "NOUN":
            if token.lemma_ in tech_keywords:
                skills.add(token.lemma_)
        if token.text in tech_keywords:
            skills.add(token.text)

    return list(skills)
import pandas as pd
from collections import Counter

def get_top_skills(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    if "skills" not in df.columns:
        print(" Колонка 'skills' отсутствует в датафрейме.")
        return pd.DataFrame()

    all_skills = [skill for sublist in df["skills"].dropna() for skill in sublist if isinstance(skill, str)]
    skill_counts = Counter(all_skills)
    top_skills = skill_counts.most_common(top_n)

    return pd.DataFrame(top_skills, columns=["Скилл", "Частота"])

