from utilities.utilities import create_json_request, save_json_to_file


deliveries_path = "./data/deliveries.jsonl"
products_path = "./data/products.jsonl"
sessions_path = "./data/sessions.jsonl"
users_path = "./data/users.jsonl"


if __name__ == "__main__":
    training, validation = create_json_request(deliveries_path, products_path, sessions_path, users_path)
    save_json_to_file("./data/requests/training_set.json", training)
    save_json_to_file("./data/requests/validation_set.json", validation)
