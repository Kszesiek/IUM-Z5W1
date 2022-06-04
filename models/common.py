from pandas import DataFrame


class Model:
    def __init__(self):
        self._model = None

    def predict(self,
                products: DataFrame,
                deliveries: DataFrame,
                sessions: DataFrame,
                users: DataFrame) -> dict[str, float]:
        # Returns prediction for input data
        raise Exception("This is an interface method")

    def generate_model(self):
        raise Exception("This is an interface method")

    def load_model(self, file_path: str):
        # Loads binary self.model from file
        self.model = None  # TODO Write reading from binary file

    @property
    def model(self):
        return self.model

    @model.setter
    def model(self, value):
        self.model = value
