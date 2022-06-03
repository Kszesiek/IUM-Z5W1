import uvicorn
from fastapi import FastAPI, Request, Response, status, HTTPException

from models import model_a, model_b

app = FastAPI()


@app.post("/predict/A")
async def get_prediction_a(request: Request, response: Response):
    data = request
    result = model_a.predict(data)
    if result:
        response.status_code = status.HTTP_200_OK
        return result
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No prediction returned.")


@app.post("/predict/B")
async def get_prediction_b(request: Request, response: Response):
    data = request
    result = model_b.predict(data)
    if result:
        response.status_code = status.HTTP_200_OK
        return result
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No prediction returned.")


@app.post("/predict")
async def get_prediction_ab(request: Request, response: Response):
    result = None
    response.status_code = status.HTTP_200_OK
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
