from datetime import datetime, timedelta

import pandas as pd
from pandas import DataFrame
from sklearn import tree

from models.common import Model


class ModelB(Model):
    def __init__(self):
        super().__init__()

    def predict(self,
                products: DataFrame,
                deliveries: DataFrame,
                sessions: DataFrame,
                users: DataFrame) -> dict[str, float]:
        # Returns prediction for input data
        pass

    def generate_model(self):
        # Calculates self.model
        self.model = None


deliveries_path = "./data/deliveries.jsonl"
products_path = "./data/products.jsonl"
sessions_path = "./data/sessions.jsonl"
users_path = "./data/users.jsonl"

deliveries_data = pd.read_json(deliveries_path, lines=True)
products_data = pd.read_json(products_path, lines=True)
sessions_data = pd.read_json(sessions_path, lines=True)
users_data = pd.read_json(users_path, lines=True)

sessions_data = pd.merge(sessions_data, products_data, on="product_id")
sessions_data = pd.merge(sessions_data, users_data, on="user_id").sort_values(by=['timestamp'], ascending=True)
# sessions_data = pd.merge(sessions_data, deliveries_data, on="purchase_id")

sessions_learn = sessions_data
sessions_validate = sessions_data

sessions_learn = sessions_learn[sessions_learn["timestamp"].dt.month != 5]
sessions_validate = sessions_validate[sessions_validate["timestamp"].dt.month == 5]
sessions_validate_period = sessions_validate["timestamp"].max().day


def convert_data_from_dataframe_to_list(dataframe_data):
    data_as_list = []
    dataframe_data = dataframe_data.drop(
        columns=['name', 'city', 'street', 'product_name', 'category_path', 'brand', 'optional_attributes',
                 'purchase_id'])
    for session in dataframe_data.iterrows():
        session = session[1]
        row = session.tolist()
        row[1] = int(round(datetime.now().timestamp() - row[1].timestamp()))
        row[4] = 1 if row[4] == 'BUY_PRODUCT' else 0
        data_as_list.append(row)

    return data_as_list


sessions_learn_list = convert_data_from_dataframe_to_list(sessions_learn)
sessions_validate_list = convert_data_from_dataframe_to_list(sessions_validate)


def prepare_labels(dataframe):
    labels = []

    for month in range(1, 5):
        expenses = {user_id: 0 for user_id in users_data["user_id"]}
        sessions_this_month = dataframe[dataframe["timestamp"].dt.month == month]

        for session in sessions_this_month.iterrows():
            session = session[1]
            user_id = session["user_id"]
            if session["event_type"] == "VIEW_PRODUCT":
                continue
            expenses[user_id] += session["price"] * (1 - session["offered_discount"] / 100)

        for session in sessions_this_month.iterrows():
            session = session[1]
            user_id = session["user_id"]
            labels.append(expenses[user_id])

    return labels


def learn(learning_data):
    y = prepare_labels(learning_data)

    fitting_data = convert_data_from_dataframe_to_list(learning_data)

    clf = tree.DecisionTreeRegressor()
    return clf.fit(fitting_data, y)


def validate(model, validating_data):
    user_predictions = {user_id: 0 for user_id in users_data["user_id"]}
    predictions_per_user = {user_id: 0 for user_id in users_data["user_id"]}

    for (user_id, prediction) in zip([row[2] for row in validating_data], model.predict(validating_data)):
        user_predictions[user_id] += prediction
        predictions_per_user[user_id] += 1

    for user_id, predictions_sum in user_predictions.items():
        # adjust to validating data - data from may ends on 25th of May, but predictions were made for a full month
        user_predictions[user_id] = predictions_sum * sessions_validate_period / 30 / (predictions_per_user[user_id] or 1)

    total_for_user = {user_id: 0 for user_id in users_data["user_id"]}

    for session in validating_data:
        if session[4] == 0:
            continue
        user_id = session[2]
        total_for_user[user_id] += session[6] * (1 - session[5] / 100)

    total = 0
    sum_ratio = 0
    for user_id in users_data["user_id"]:
        if total_for_user[user_id] == 0:
            ratio = 0 if user_predictions[user_id] < 10 else min(user_predictions[user_id] / 10, 1)
        else:
            ratio = min((user_predictions[user_id] - total_for_user[user_id]) / total_for_user[user_id], 1)
        print(f"user {user_id}: {ratio:3.2f} ({user_predictions[user_id]:3.1f}/{total_for_user[user_id]:3.1f})")
        total += total_for_user[user_id]
        sum_ratio += abs(ratio)
    print(f"dokładność: {100 * (1 - sum_ratio / len(users_data.index)):6.2f}%")
    print(f"suma modelu: {sum(user_predictions.values()):9.2f}")
    print(f"suma actual: {total:9.2f}")

    # dokładność: 36.49 %
    # suma modelu: 247519.09
    # suma actual: 219801.26


def predict(data):
    return None


if __name__ == "__main__":
    model = learn(sessions_learn)
    validate(model, sessions_validate_list)
