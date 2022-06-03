import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/predict/A")
async def get_prediction_a(request: Request):
    result = None
    return result


@app.post("/predict/B")
async def get_prediction_b(request: Request):
    result = None
    return result


@app.post("/predict")
async def get_prediction_ab(request: Request):
    result = None
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
