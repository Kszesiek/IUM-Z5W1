import functools
import json
import pandas as pd

from random import shuffle
from fastapi import status, HTTPException

from models.common import ModelNotInitialisedException

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


def split_data(data):
    shuffle(data)
    half_size = len(data) // 2
    subset_1 = data[:half_size]
    subset_2 = data[half_size:]

    return subset_1, subset_2


def convert_to_dataframe(data):
    products = pd.DataFrame([vars(product) for product in data.products])
    deliveries = pd.DataFrame([vars(delivery) for delivery in data.deliveries])
    sessions = pd.DataFrame([vars(session) for session in data.sessions])
    users = pd.DataFrame([vars(user) for user in data.users])
    return products, deliveries, sessions, users


def catch_exceptions(function):
    """A general decorator function to catch and signalise model exceptions"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except ModelNotInitialisedException:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="The model has not been initialised.")
        return result

    return wrapper


if __name__ == "__main__":
    json_request = create_json_request()
    save_json_to_file("./data/requests/data_request.json", json_request)
