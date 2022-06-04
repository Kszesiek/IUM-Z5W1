from datetime import datetime
from os.path import exists

import uvicorn
from fastapi import FastAPI, Request, Response, status, HTTPException
from pydantic import BaseModel, Field

from models.model_a import ModelA
from models.model_b_simplified import ModelB

from utilities.utilities import convert_to_dataframe, split_data

model_a_model_path = "./data/binary_model_a.bin"
model_b_model_path = "./data/binary_model_b.bin"
model_a: ModelA = ModelA(model_a_model_path)
model_b: ModelB = ModelB(model_b_model_path)

app = FastAPI()


class DeliveryModel(BaseModel):
    purchase_id: int
    purchase_timestamp: datetime
    delivery_timestamp: datetime
    delivery_company: int


class ProductModel(BaseModel):
    product_id: int
    product_name: str
    category_path: str
    price: float = Field(...)
    brand: str
    weight_kg: float
    optional_attributes: dict[str, str] = Field(...)


class SessionModel(BaseModel):
    session_id: int
    timestamp: datetime
    user_id: int
    product_id: int
    event_type: str
    offered_discount: int
    purchase_id: int = None


class UserModel(BaseModel):
    user_id: int
    name: str
    city: str
    street: str


class PredictRequestModel(BaseModel):
    deliveries: list[DeliveryModel]
    products: list[ProductModel]
    sessions: list[SessionModel]
    users: list[UserModel]


@app.on_event("startup")
async def startup_event():
    for model in (model_a, model_b):
        if model.model is None:
            if exists(model.file_path):
                model.load_model_from_file()


@app.post("/predict/A")
async def get_prediction_a(body: PredictRequestModel, request: Request, response: Response):
    products, deliveries, sessions, users = convert_to_dataframe(body)
    result = model_a.predict(products, deliveries, sessions, users)
    if result:
        response.status_code = status.HTTP_200_OK
        return result
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction returned.")


@app.post("/predict/B")
async def get_prediction_b(body: PredictRequestModel, response: Response):
    products, deliveries, sessions, users = convert_to_dataframe(body)
    result = model_b.predict(products, deliveries, sessions, users)
    if result:
        response.status_code = status.HTTP_200_OK
        return result
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction returned.")


@app.post("/predict")
async def get_prediction_ab(request: Request, response: Response):
    result = None
    response.status_code = status.HTTP_200_OK
    return result


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
