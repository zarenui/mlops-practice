from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import os

# 1. Определяем абсолютные пути (чтобы избежать ошибок при деплое)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

# 2. Загружаем модель ОДИН раз при старте сервера! (Best Practice)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris ML API", description="API для предсказания сорта Ириса")


# 3. Редиректим нас сразу на главную страницу.
@app.get("/")
def read_root():
    # Автоматически перенаправляем пользователя на страницу с документацией
    return RedirectResponse(url="/docs")


# 4. Контракт данных (Pydantic). FastAPI не пропустит запросы с неверными типами.
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# 5. Эндпоинт для проверки здоровья сервера (Uptime Monitor)
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running"}


# 6. Эндпоинт для предсказания (принимает JSON, возвращает JSON)
@app.post("/predict")
def predict(features: IrisFeatures):
    # Преобразуем входящие данные в массив для scikit learn
    data = [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]
    ]
    prediction = model.predict(data)

    # Возвращаем результат
    return {"predicted_class": int(prediction[0])}
