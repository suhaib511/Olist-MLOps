from pydantic import BaseModel

class PredictionRequest(BaseModel):
    purchase_year: int
    purchase_month: int
    purchase_dayofweek: int
    approval_delay_hours: float | None
    estimated_delivery_days: float
    count_item: float
    total_price: float
    total_freight: float
    count_sequential: float | None
    sum_pyament_value: float | None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    instances: list[PredictionRequest]