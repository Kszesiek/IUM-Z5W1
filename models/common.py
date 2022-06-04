import pandas as pd
from pandas import DataFrame


class Model:
    def __init__(self, file_path: str):
        self._model = None
        self._file_path: str = file_path

    def predict(self,
                products: DataFrame,
                deliveries: DataFrame,
                sessions: DataFrame,
                users: DataFrame) -> dict[str, float]:
        # Returns prediction for input data
        raise Exception("This is an interface method")

    def generate_model(self,
                       products: DataFrame,
                       deliveries: DataFrame,
                       sessions: DataFrame,
                       users: DataFrame):
        raise Exception("This is an interface method")

    def load_model_from_file(self):
        raise Exception("This is an interface method")

    def save_model_to_file(self):
        raise Exception("This is an interface method")

    def verify(self,
               products: DataFrame,
               deliveries: DataFrame,
               sessions: DataFrame,
               users: DataFrame):
        # Verifies how good is the model

        predictions = self.predict(products, deliveries, sessions, users)

        sessions = pd.merge(sessions, products, on="product_id")
        sessions = pd.merge(sessions, users, on="user_id").sort_values(by=['timestamp'], ascending=True)

        total_for_user = {user_id: 0 for user_id in users["user_id"]}

        for session in sessions.iterrows():
            session = session[1]
            user_id = session["user_id"]
            if session["event_type"] == "VIEW_PRODUCT":
                continue
            total_for_user[user_id] += session["price"] * (1 - session["offered_discount"] / 100)

        total = 0
        sum_ratio = 0
        for user_id in users["user_id"]:
            if total_for_user[user_id] == 0:
                ratio = 0 if predictions[user_id] < 10 else min(predictions[user_id] / 10, 1)
            else:
                ratio = (predictions[user_id] - total_for_user[user_id]) / max(predictions[user_id], total_for_user[user_id])

            print(f"user {user_id}: {ratio:3.2f} (expected: {total_for_user[user_id]:3.1f}, got: {predictions[user_id]:3.1f})")
            total += total_for_user[user_id]
            sum_ratio += abs(ratio)
        print(f"dokładność: {100 * (1 - sum_ratio / len(users.index)):6.2f}%")
        print(f"suma modelu: {sum(predictions.values()):9.2f}")
        print(f"suma actual: {total:9.2f}")

        return 1 - sum_ratio / len(users.index)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value


class ModelException(Exception):
    def __init__(self, message: str = ""):
        super(ModelException, self).__init__(message)


class ModelNotInitialisedException(Exception):
    def __init__(self, message: str = ""):
        super(ModelNotInitialisedException, self).__init__(message)
