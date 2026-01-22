import pandas as pd
import joblib
import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from src.pipeline_dados import preparar_dados_para_treino, criar_pre_processador
from src.pipeline_modelos import treinar_e_salvar

app = FastAPI(title="API Preço Apartamentos SP", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem (ou coloque a URL do Lovable aqui)
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Permite qualquer cabeçalho
)

# --- Configurações ---
CAMINHO_HISTORICO = "dados/historico_apartamentos.csv"
CAMINHO_MODELO = "artifacts/modelo_campeao.pkl"

# Garante criação das pastas
os.makedirs(os.path.dirname(CAMINHO_HISTORICO), exist_ok=True)
os.makedirs(os.path.dirname(CAMINHO_MODELO), exist_ok=True)

modelo_carregado = None

# Input continua minúsculo (Snake Case) pois é o padrão de APIs JSON
class ApartamentoInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    parking_spaces: int
    neighborhood: str
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    created_date: Optional[int] = 2024 # Valor default

@app.on_event("startup")
def carregar_modelo():
    global modelo_carregado
    if os.path.exists(CAMINHO_MODELO):
        try:
            modelo_carregado = joblib.load(CAMINHO_MODELO)
            print(f"Modelo carregado com sucesso!")
        except:
            print("Erro ao carregar modelo.")
    else:
        print("Aviso: Nenhum modelo encontrado. Utilize /train.")

@app.post("/train")
async def endpoint_train(file: UploadFile = File(...)):
    global modelo_carregado
    filename = file.filename.lower()
    contents = await file.read()
    
    try:
        if filename.endswith(".csv"):
            df_novo = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            df_novo = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Formato inválido.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro leitura: {str(e)}")

    try:
        X, y = preparar_dados_para_treino(df_novo, CAMINHO_HISTORICO)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro dados: {str(e)}")

    try:
        preprocessor = criar_pre_processador()
        score = treinar_e_salvar(X, y, preprocessor, CAMINHO_MODELO)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erro treino: {str(e)}")
    
    # Recarrega o modelo para garantir que está usando a versão nova
    modelo_carregado = joblib.load(CAMINHO_MODELO)
    
    return {
        "status": "Sucesso",
        "metricas_R2": round(float(score), 4), # Convertemos score também por segurança
        "colunas_usadas": list(X.columns)
    }

@app.post("/predict")
def endpoint_predict(features: ApartamentoInput):
    global modelo_carregado
    
    if not modelo_carregado:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")
    
    # 1. Cria DataFrame com nomes minúsculos (do Pydantic)
    # Se estiver usando Pydantic v2, features.model_dump() é preferível, mas dict() funciona
    try:
        dados = features.model_dump()
    except AttributeError:
        dados = features.dict()
        
    df_input = pd.DataFrame([dados])
    
    # 2. RENOMEIA para Maiúsculo (O que o modelo espera)
    mapa_para_modelo = {
        'area': 'Area',
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms',
        'parking_spaces': 'Parking_Spaces',
        'neighborhood': 'Neighborhood',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'created_date': 'Created_Date'
    }
    df_input = df_input.rename(columns=mapa_para_modelo)
    
    try:
        # --- CORREÇÃO DO ERRO AQUI ---
        # O modelo retorna numpy.float32, o FastAPI não entende.
        # Envolvemos em float() para converter para Python nativo.
        predicao_numpy = modelo_carregado.predict(df_input)[0]
        predicao = float(predicao_numpy)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro predição: {str(e)}")
        
    return {
        "preco_estimado": round(predicao, 2),
        "neighborhood": features.neighborhood
    }