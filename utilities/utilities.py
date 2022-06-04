import json

import pandas as pd

deliveries_path = "./data/deliveries.jsonl"
products_path = "./data/products.jsonl"
sessions_path = "./data/sessions.jsonl"
users_path = "./data/users.jsonl"


def create_json_request():
    deliveries_data = pd.read_json(deliveries_path, lines=True)
    products_data = pd.read_json(products_path, lines=True)
    sessions_data = pd.read_json(sessions_path, lines=True)
    users_data = pd.read_json(users_path, lines=True)

    users = json.loads(users_data.to_json(orient='records'))
    sessions = json.loads(sessions_data.to_json(orient='records', date_unit='ms'))
    products = json.loads(products_data.to_json(orient='records'))
    deliveries = json.loads(deliveries_data.to_json(orient='records'))
    json_object = {"users": users, "sessions": sessions, "products": products, "deliveries": deliveries}

    return json_object


def save_json_to_file(file_name, json_object):
    with open(file_name, "w+", encoding="utf-8") as file:
        file.write(json.dumps(json_object, indent=4, sort_keys=True))


if __name__ == "__main__":
    json_request = create_json_request()
    save_json_to_file("./data/requests/data_request.json", json_request)
