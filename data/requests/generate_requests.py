from utilities.utilities import create_json_request, save_json_to_file

if __name__ == "__main__":
    training, validation = create_json_request()
    save_json_to_file("./data/requests/training_set.json", training)
    save_json_to_file("./data/requests/validation_set.json", validation)
