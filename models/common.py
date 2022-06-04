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
        with open(self._file_path, "rb") as file:
            self._model = file.read()

    def save_model_to_file(self):
        with open(self._file_path, "rb") as file:
            # self._model = file.write()
            pass

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
