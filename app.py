import pandas as pd
import joblib
import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from src.pipeline_dados import preparar_dados_para_treino, criar_pre_processador
from src.pipeline_modelo import treinar_e_salvar

app = FastAPI(title="API Preço Apartamentos SP", version="1.0")

# --- Configurações ---
CAMINHO_HISTORICO = "src/dados/historico_apartamentos.csv"
CAMINHO_MODELO = "src/artifacts/modelo_campeao.pkl"

modelo_carregado = None

class ApartamentoInput(BaseModel):
    area: float
    rooms: int
    bathroom: int
    parking_spaces: int
    condo: Optional[float] = 0
    hoa: Optional[float] = 0
    neighborhood: str
    district: str

@app.on_event("startup")
def carregar_modelo():
    global modelo_carregado
    if os.path.exists(CAMINHO_MODELO):
        modelo_carregado = joblib.load(CAMINHO_MODELO)
        print("Modelo carregado com sucesso!")
    else:
        print("Aviso: Nenhum modelo encontrado. Utilize o endpoint /train.")

# --- NOVO ENDPOINT DE TREINO (Upload de Arquivo) ---
@app.post("/train")
async def endpoint_train(file: UploadFile = File(...)):
    """
    Recebe um arquivo CSV ou Excel (.xlsx), processa os dados,
    atualiza o histórico e re-treina o modelo.
    """
    global modelo_carregado
    
    # 1. Verificar extensão do arquivo
    filename = file.filename.lower()
    contents = await file.read() # Lê o arquivo da memória
    
    try:
        if filename.endswith(".csv"):
            # Lê CSV a partir dos bytes
            df_novo = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            # Lê Excel a partir dos bytes
            df_novo = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Formato inválido. Envie .csv ou .xlsx")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler o arquivo: {str(e)}")

    # 2. Validação Básica: O arquivo tem a coluna alvo 'price'?
    if 'price' not in df_novo.columns:
        raise HTTPException(status_code=400, detail="O arquivo de treino DEVE conter a coluna 'price'.")

    # 3. Pipeline de Dados (Carga + Limpeza + Concatenação)
    # Nota: Aqui reutilizamos sua lógica existente
    try:
        X, y = preparar_dados_para_treino(df_novo, CAMINHO_HISTORICO)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento dos dados: {str(e)}")
    
    # 4. Pipeline de Modelo (Treino + Geração de Binário)
    try:
        preprocessor = criar_pre_processador()
        score = treinar_e_salvar(X, y, preprocessor, CAMINHO_MODELO)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erro no treinamento do modelo: {str(e)}")
    
    # 5. Hot Reload: Atualizar modelo na memória da API
    modelo_carregado = joblib.load(CAMINHO_MODELO)
    
    return {
        "status": "Sucesso",
        "mensagem": "Dataset atualizado e modelo re-treinado.",
        "arquivo_recebido": file.filename,
        "total_linhas_processadas": len(X),
        "performance_r2": round(score, 4)
    }

@app.post("/predict")
def endpoint_predict(features: ApartamentoInput):
    global modelo_carregado
    
    if not modelo_carregado:
        raise HTTPException(status_code=503, detail="Modelo não disponível. Faça o upload de dados em /train primeiro.")
    
    df_input = pd.DataFrame([features.dict()])
    
    try:
        predicao = modelo_carregado.predict(df_input)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
        
    return {
        "preco_estimado": round(predicao, 2)
    }