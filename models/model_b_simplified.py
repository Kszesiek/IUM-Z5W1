import pickle
from datetime import datetime

import pandas as pd
from pandas import DataFrame
from sklearn import tree

from models.common import Model, ModelNotInitialisedException
from utilities.utilities import catch_exceptions


class ModelB(Model):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    @catch_exceptions
    def predict(self,
                products: DataFrame,
                deliveries: DataFrame,
                sessions: DataFrame,
                users: DataFrame) -> dict[str, float]:
        # Returns prediction for input data
        if not self._model:
            raise ModelNotInitialisedException("Model has not been initialised.")

        sessions = pd.merge(sessions, products, on="product_id")
        sessions = pd.merge(sessions, users, on="user_id").sort_values(by=['timestamp'], ascending=True)
        sessions_list = self.convert_data_from_dataframe_to_list(sessions)

        user_predictions = {user_id: 0 for user_id in users["user_id"]}
        predictions_per_user = {user_id: 0 for user_id in users["user_id"]}

        for (user_id, prediction) in zip([row[2] for row in sessions_list], self.model.predict(sessions_list)):
            if user_id in users["user_id"].values:
                user_predictions[user_id] += prediction
                predictions_per_user[user_id] += 1

        for user_id, predictions_sum in user_predictions.items():
            user_predictions[user_id] = predictions_sum / (predictions_per_user[user_id] or 1)

        return user_predictions

    def generate_model(self,
                       products: DataFrame,
                       deliveries: DataFrame,
                       sessions: DataFrame,
                       users: DataFrame):
        # Calculates self.model

        sessions = pd.merge(sessions, products, on="product_id")
        sessions = pd.merge(sessions, users, on="user_id").sort_values(by=['timestamp'], ascending=True)

        y = self.prepare_labels(sessions)
        fitting_data = self.convert_data_from_dataframe_to_list(sessions)
        clf = tree.DecisionTreeRegressor()
        self.model = clf.fit(fitting_data, y)

    def load_model_from_file(self):
        with open(self.file_path, "rb") as file:
            self._model = pickle.load(file)

    def save_model_to_file(self):
        with open(self._file_path, "wb") as file:
            pickle.dump(self._model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def convert_data_from_dataframe_to_list(self, dataframe_data):
        data_as_list = []
        dataframe_data = dataframe_data.drop(
            columns=['name', 'city', 'street', 'product_name', 'category_path', 'brand', 'optional_attributes', 'purchase_id'])
        for session in dataframe_data.iterrows():
            session = session[1]
            row = session.tolist()
            row[1] = int(round(datetime.now().timestamp() - row[1].timestamp()))
            row[4] = 1 if row[4] == 'BUY_PRODUCT' else 0
            data_as_list.append(row)

        return data_as_list

    def prepare_labels(self, dataframe):
        labels = []

        min_month = dataframe["timestamp"].min().month
        max_month = dataframe["timestamp"].max().month

        for month in range(min_month, max_month + 1):
            expenses = {user_id: 0 for user_id in dataframe["user_id"]}
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


if __name__ == "__main__":
    deliveries_path = "./data/deliveries.jsonl"
    products_path = "./data/products.jsonl"
    sessions_path = "./data/sessions.jsonl"
    users_path = "./data/users.jsonl"

    deliveries_data = pd.read_json(deliveries_path, lines=True)
    products_data = pd.read_json(products_path, lines=True)
    sessions_data = pd.read_json(sessions_path, lines=True)
    users_data = pd.read_json(users_path, lines=True)

    sessions_learn = sessions_data[sessions_data["timestamp"].dt.month != 5]
    sessions_verify = sessions_data[sessions_data["timestamp"].dt.month == 5]

    modelB = ModelB("./data/binary_model_b.bin")
    modelB.generate_model(products_data, deliveries_data, sessions_learn, users_data)
    print(modelB.verify(products_data, deliveries_data, sessions_verify, users_data))
