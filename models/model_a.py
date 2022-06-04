import math
from datetime import timedelta, datetime
import pandas as pd
from pandas import DataFrame

from microservice.api import model_a_model_path
from models.common import Model, ModelNotInitialisedException
from utilities.utilities import catch_exceptions


class ModelA(Model):
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

        return {user_id: self.model[user_id] for user_id in users["user_id"]}

    def generate_model(self,
                       products: DataFrame,
                       deliveries: DataFrame,
                       sessions: DataFrame,
                       users: DataFrame):
        # Calculates self.model

        sessions = pd.merge(sessions, products, on="product_id").drop(
            columns=['product_name', 'category_path']).sort_values(by=['timestamp'], ascending=False)

        next_purchase = {user_id: datetime.now() for user_id in users["user_id"]}
        spending_factors = {user_id: 0 for user_id in users["user_id"]}
        actual_total = 0

        learning_days = (max(sessions["timestamp"]) - min(sessions["timestamp"])).days
        last_session_timestamp = sessions["timestamp"].max()

        for session in sessions.iterrows():
            session = session[1]
            user_id = session["user_id"]
            if session["event_type"] == "VIEW_PRODUCT":
                continue
            actual_total += session["price"] * (1 - session["offered_discount"] / 100)
            if next_purchase[user_id]:
                time_difference = max(next_purchase[user_id] - session["timestamp"], timedelta(hours=24))
                time_difference = time_difference.total_seconds() / 86400
                days_from_today = (last_session_timestamp - session["timestamp"]).total_seconds() / 86400 / 30
                favor_newer_records_with_tanh = (0.5 + 0.5 * math.tanh(- days_from_today / 2 + 2))
                spending_factors[user_id] += session["price"] * (
                        1 - session["offered_discount"] / 100) / time_difference * favor_newer_records_with_tanh
            next_purchase[user_id] = session["timestamp"]

        correction = actual_total / sum(spending_factors.values())
        spending_factors = {user_id: factor * correction for (user_id, factor) in spending_factors.items()}

        max_factor = max(spending_factors.values())
        spending_factors_perc = {key: val / max_factor * 100 for (key, val) in spending_factors.items()}
        self.model = {key: val * 30 / learning_days for (key, val) in spending_factors.items()}


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

    modelA = ModelA(model_a_model_path)
    modelA.generate_model(products_data, deliveries_data, sessions_data, users_data)
    print(modelA.verify(products_data, deliveries_data, sessions_verify, users_data))
