import pandas as pd
import joblib
import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

# --- Importação dos Módulos do Pipeline ---
# Ajustado para importar de 'src.pipeline_modelos' (plural) conforme seu arquivo
from src.pipeline_dados import preparar_dados_para_treino, criar_pre_processador
from src.pipeline_modelos import start

app = FastAPI(title="API Preço Apartamentos SP", version="1.0")

# --- Configurações de Caminhos ---
CAMINHO_HISTORICO = "./dados/dados_tratados.csv"
CAMINHO_MODELO = "./artifacts/melhor_modelo.pkl"
os.makedirs(os.path.dirname(CAMINHO_HISTORICO), exist_ok=True)
os.makedirs(os.path.dirname(CAMINHO_MODELO), exist_ok=True)

# Variável global para manter o modelo em memória (Hot Reload)
modelo_carregado = None

# --- Schema de Entrada (Pydantic) ---
# Atualizado para bater com os nomes gerados pelo pipeline_dados.py
class ApartamentoInput(BaseModel):
    area: float            # Mapeado de 'Area'
    bedrooms: int          # Mapeado de 'Bedrooms'
    bathrooms: int         # Mapeado de 'Bathrooms'
    parking_spaces: int    # Mapeado de 'Parking_Spaces'
    neighborhood: str      # Gerado pelo tratamento de endereço
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    # price não entra aqui pois é o target

# --- Evento de Inicialização ---
@app.on_event("startup")
def carregar_modelo():
    """Carrega o modelo na memória ao iniciar a API"""
    global modelo_carregado
    if os.path.exists(CAMINHO_MODELO):
        try:
            modelo_carregado = joblib.load(CAMINHO_MODELO)
            print(f"Modelo carregado com sucesso de: {CAMINHO_MODELO}")
        except Exception as e:
            print(f"Erro ao carregar modelo existente: {e}")
    else:
        print("Aviso: Nenhum modelo encontrado. Utilize o endpoint /train para criar o primeiro.")

# --- ENDPOINT DE TREINO (Pipeline Completo) ---
@app.post("/train")
async def endpoint_train(file: UploadFile = File(...)):
    """
    Recebe arquivo (CSV/Excel), roda o pipeline de dados + pipeline de modelos
    e atualiza a API em tempo real.
    """
    print("Iniciando o processo de treino via endpoint /train...")
    global modelo_carregado
    
    # 1. Leitura do Arquivo
    filename = file.filename.lower()
    contents = await file.read()
    
    try:
        if filename.endswith(".csv"):
            df_novo = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            df_novo = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Formato inválido. Envie .csv ou .xlsx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {str(e)}")

    # 2. Execução do Pipeline de DADOS
    # Limpa, concatena com histórico e separa X (features) e y (target)
    try:
        X, y = preparar_dados_para_treino(df_novo, CAMINHO_HISTORICO)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Pipeline de Dados: {str(e)}")

    # 3. Execução do Pipeline de MODELOS
    # Cria a receita de pré-processamento e treina o modelo
    try:
        preprocessor = criar_pre_processador()
        
        # Chama a função de treino que orquestra tudo e salva o .pkl
        # Retorna uma métrica (ex: MAE ou R2) para feedback
        score = start(CAMINHO_HISTORICO)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Pipeline de Modelos: {str(e)}")

    # 4. Hot Reload (Recarregar o modelo novo na memória)
    try:
        modelo_carregado = joblib.load(CAMINHO_MODELO)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recarregar o modelo treinado: {str(e)}")

    return {
        "status": "Sucesso",
        "mensagem": "Dataset histórico atualizado e modelo re-treinado.",
        "arquivo_processado": file.filename,
        "amostras_treinamento": len(X),
        "metricas_performance": score 
    }

# --- ENDPOINT DE PREDIÇÃO ---
@app.post("/predict")
def endpoint_predict(features: ApartamentoInput):
    """
    Recebe características do imóvel e retorna o preço estimado.
    """
    global modelo_carregado
    
    if not modelo_carregado:
        raise HTTPException(status_code=503, detail="Modelo indisponível. Treine primeiro via /train.")
    
    # Converte o input Pydantic para DataFrame
    df_input = pd.DataFrame([features.dict()])
    
    try:
        # O pipeline salvo no .pkl já contém o pré-processador.
        # Ele vai tratar o 'neighborhood' e escalar os números automaticamente.
        predicao = modelo_carregado.predict(df_input)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
        
    return {
        "bairro": features.neighborhood,
        "preco_estimado": round(predicao, 2),
        "moeda": "BRL"
    }