def analyze_data(data):
    grouped_data = data.groupby("area").size().reset_index(name="Количество вакансий")
    return grouped_data
