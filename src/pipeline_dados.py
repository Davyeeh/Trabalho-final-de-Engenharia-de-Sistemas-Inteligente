import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- CONFIGURAÇÃO (MAIÚSCULAS) ---
# O modelo agora espera colunas começando com Maiúscula
COLUNAS_NUMERICAS = ['Area', 'Bedrooms', 'Bathrooms', 'Parking_Spaces', 'Latitude', 'Longitude', 'Created_Date']
COLUNAS_CATEGORICAS = ['Neighborhood'] 
TARGET = 'Price'

def _separar_endereco(texto):
    """
    Função auxiliar para quebrar o endereço em Rua e Bairro.
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return pd.Series(["Não Informado", "Não Informado"])
    
    try:
        parte1, _ = texto.split(" - ", 1)
    except ValueError:
        parte1 = texto

    if "," in parte1:
        rua, bairro = parte1.rsplit(",", 1)
    else:
        rua = "Não Informado"
        bairro = parte1
        
    return pd.Series([rua.strip(), bairro.strip()])

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza e Padronização para nomes com Maiúscula.
    """
    df = df.copy()

    # 1. Tratamento de Nulos na Localização
    cols_loc = ['Latitude', 'Longitude']
    if all(c in df.columns for c in cols_loc):
        df = df.dropna(subset=cols_loc)

    # 2. Ajuste do Preço
    if 'Price' in df.columns:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').astype(float)
        precos_validos = df.loc[df['Price'] > 0, 'Price']
        if not precos_validos.empty:
            Q1 = precos_validos.quantile(0.25)
            Q3 = precos_validos.quantile(0.75)
            IQR = Q3 - Q1
            limite = Q3 + 1.5 * IQR
            media_justa = precos_validos[precos_validos <= limite].mean()
            mask_zero = (df['Price'] == 0) | (df['Price'].isna())
            df.loc[mask_zero, 'Price'] = float(media_justa)

    # 3. Remoção de Duplicatas
    df = df.drop_duplicates()

    # 4. Feature Engineering: Endereço (Cria Neighborhood)
    if 'Adress' in df.columns:
        # Usamos nomes temporários aqui, mas depois renomeamos
        df[['Street_Temp', 'Neighborhood']] = df['Adress'].apply(_separar_endereco)
        df = df.drop(columns=['Adress', 'Street_Temp'], errors='ignore')

    # 5. Feature Engineering: Data (Extrair Ano)
    if 'created_date' in df.columns:
        df['created_date'] = df['created_date'].astype(str).str[:4].astype(int)

    # 6. RENOMEAÇÃO (PADRONIZAR PARA MAIÚSCULA)
    # Garante que created_date vire Created_Date e outros nomes fiquem certos
    mapa_nomes = {
        'created_date': 'Created_Date',
        'neighborhood': 'Neighborhood',
        'area': 'Area',
        'price': 'Price',
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms',
        'parking_spaces': 'Parking_Spaces',
        'latitude': 'Latitude',
        'longitude': 'Longitude'
    }
    df = df.rename(columns=mapa_nomes)

    # 7. Remoção de colunas inúteis
    colunas_remover = ['ID', 'below_price', 'extract_date', 'Rua', 'Bairro', 'street', 'Street'] 
    df = df.drop(columns=colunas_remover, errors='ignore')

    return df

def preparar_dados_para_treino(df_novo, caminho_historico):
    df_limpo = limpar_dados(df_novo)
    
    try:
        df_historico = pd.read_csv(caminho_historico)
        df_full = pd.concat([df_historico, df_limpo], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_full = df_limpo
    
    df_full.to_csv(caminho_historico, index=False)
    
    cols_existentes = [c for c in COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS if c in df_full.columns]
    
    if TARGET not in df_full.columns:
        raise ValueError(f"A coluna alvo '{TARGET}' não foi encontrada.")

    X = df_full[cols_existentes]
    y = df_full[TARGET]
    
    return X, y

def criar_pre_processador():
    # Pipeline configurado para nomes Maiúsculos
    features_num = ['Area', 'Bedrooms', 'Bathrooms', 'Parking_Spaces', 'Latitude', 'Longitude', 'Created_Date']
    features_cat = ['Neighborhood']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_num),
            ('cat', categorical_transformer, features_cat)
        ])
    
    return preprocessor