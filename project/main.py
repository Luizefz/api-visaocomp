import requests
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# Inicialize o FastAPI

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*",  # Permitir todas as origens
    # "http://localhost",
    # "http://localhost:8000",
    # Adicione outras origens permitidas aqui
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permitir apenas origens especificadas
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos HTTP
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Carregar o modelo de detecção EfficientDet (feito apenas uma vez ao iniciar)
with open('project/efficientdet_lite0.tflite', 'rb') as f:
    model = f.read()

# Criar o detector de objetos
base_options = python.BaseOptions(model_asset_buffer=model)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Definir o modelo de requisição para a API (URL da imagem)
class ImageRequest(BaseModel):
    image_url: str

# Função para carregar a imagem a partir da URL
def load_image_from_url(image_url: str) -> np.ndarray:
    """
    Função para carregar uma imagem de uma URL e convertê-la para o formato adequado.
    """
    response = requests.get(image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Não foi possível baixar a imagem da URL fornecida.")
    
    # Abrir a imagem usando PIL
    img = Image.open(io.BytesIO(response.content))
    img = img.convert("RGB")  # Garantir que seja no formato RGB
    return img

# Função para detectar objetos na imagem usando o MediaPipe
def detect_objects(image: Image) -> list:
    """
    Função para processar a imagem e detectar os objetos.
    """
    # Lista de objetos que queremos detectar
    target_objects = ['laptop', 'mouse', 'remote', 'keyboard', 'cell phone']

    # Converter a imagem PIL para um array numpy
    img_array = np.array(image)

    # Converter a imagem numpy para um formato que o MediaPipe aceite (imagem RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    # Detectar os objetos
    detection_result = detector.detect(mp_image)

    # Extrair os dados das detecções e filtrar apenas os objetos presentes na lista target_objects
    detection_data = [
        category.category_name 
        for detection in detection_result.detections 
        for category in detection.categories 
        if category.category_name in target_objects
    ]
    
    return detection_data


@app.post("/predict/")
async def predict_device(request: ImageRequest):
    try:
        # Carregar a imagem da URL fornecida
        image = load_image_from_url(request.image_url)

        # Detectar os objetos na imagem
        detections = detect_objects(image)

        # Retornar as deteções em um formato legível
        if detections:
            return {"devices": detections}
        else:
            raise HTTPException(status_code=404, detail="Nenhum objeto detectado na imagem.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {str(e)}")
