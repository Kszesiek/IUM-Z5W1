from datetime import datetime
from pydantic import BaseModel, Field


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
