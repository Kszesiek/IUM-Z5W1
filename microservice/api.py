from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, Response, status, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List

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


@app.post("/predict/A")
async def get_prediction_a(body: PredictRequestModel, response: Response):
    # result = model_a.predict(body)
    result = body
    if result:
        response.status_code = status.HTTP_200_OK
        return result
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction returned.")


@app.post("/predict/B")
async def get_prediction_b(body: PredictRequestModel, response: Response):
    # result = model_b_simplified.predict(request)
    result = body
    if result:
        response.status_code = status.HTTP_200_OK
        return result
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction returned.")


@app.post("/predict")
async def get_prediction_ab(request: Request, response: Response):
    result = None
    response.status_code = status.HTTP_200_OK
    return result


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
