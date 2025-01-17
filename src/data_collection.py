import requests

def fetch_all_vacancies(query="аналитик данных", area=1):
    url = "https://api.hh.ru/vacancies"
    all_vacancies = []
    page = 0
    max_pages = 3


    while True:
        params = {
            "text": query,
            "area": area,
            "per_page": 100,
            "page": page
        }
        response = requests.get(url, params=params)

        if response.status_code != 200:
            break

        data = response.json()
        vacancies = data.get("items", [])
        all_vacancies.extend(vacancies)

        print(f"Собрано {len(vacancies)} вакансий с {page + 1}-й страницы")

        if len(vacancies) == 0 or page >= data.get("pages", 1) - 1:
            break

        page += 1
        if page >= max_pages:
            print("Достигнут лимит страниц")
            break

    print(f"Всего собрано {len(all_vacancies)} вакансий")
    return all_vacancies


vacancies = fetch_all_vacancies()
