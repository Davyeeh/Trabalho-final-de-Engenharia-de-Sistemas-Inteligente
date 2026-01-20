import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- CONFIGURAÇÃO CORRIGIDA ---
# Agora as listas usam os nomes PÓS-RENOMEAÇÃO (minúsculos/inglês)
# Isso corrige o erro de "not in index"
COLUNAS_NUMERICAS = ['area', 'bedrooms', 'bathrooms', 'parking_spaces', 'latitude', 'longitude', 'price']
COLUNAS_CATEGORICAS = ['neighborhood'] 
TARGET = 'Price'

def _separar_endereco(texto):
    """Auxiliar para separar Rua e Bairro."""
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
    """Limpeza completa e Renomeação."""
    df = df.copy()

    # 1. Tratamento de Nulos na Localização
    cols_loc = ['Latitude', 'Longitude']
    if all(c in df.columns for c in cols_loc):
        df = df.dropna(subset=cols_loc)

    # 2. Ajuste do Preço (IQR)
    if 'Price' in df.columns:
        # Força conversão para float
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

    # 4. Ajuste no Endereço
    if 'Adress' in df.columns:
        df[['street', 'neighborhood']] = df['Adress'].apply(_separar_endereco)
        df = df.drop(columns=['Adress'], errors='ignore')

    # 5. Ajuste de Data
    if 'created_date' in df.columns:
        df['created_date'] = df['created_date'].astype(str).str.split('T').str[0]

    # 6. Remoção de colunas inúteis
    colunas_remover = ['ID', 'below_price', 'extract_date', 'Rua', 'Bairro'] 
    df = df.drop(columns=colunas_remover, errors='ignore')

    # 7. RENOMEAÇÃO FINAL (CamelCase -> snake_case)
    # É aqui que 'Area' vira 'area', etc.
    # mapa_nomes = {
    #     'Area': 'area',
    #     'Bedrooms': 'bedrooms',
    #     'Bathrooms': 'bathrooms',
    #     'Parking_Spaces': 'parking_spaces',
    #     'Price': 'price',
    #     'Latitude': 'latitude',
    #     'Longitude': 'longitude',
    # }
    # df = df.rename(columns=mapa_nomes)

    return df

def preparar_dados_para_treino(df_novo, caminho_historico):
    # 1. Limpa os dados novos
    df_limpo = limpar_dados(df_novo)
    
    # 2. Concatena com histórico
    try:
        df_historico = pd.read_csv(caminho_historico)
        df_full = pd.concat([df_historico, df_limpo], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_full = df_limpo
    
    df_full.to_csv(caminho_historico, index=False)
    
    # 3. Seleção de Features
    # Agora usamos os nomes corretos (snake_case) para filtrar
    cols_existentes = [c for c in COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS if c in df_full.columns]
    
    # Validação do Target
    if TARGET not in df_full.columns:
        # Fallback caso a renomeação tenha falhado em algum ponto
        raise ValueError(f"Coluna alvo '{TARGET}' não encontrada. Colunas disponíveis: {df_full.columns.tolist()}")

    X = df_full[cols_existentes]
    y = df_full[TARGET]
    
    return X, y

def criar_pre_processador():
    # Define as transformações usando os nomes NOVOS
    features_num = ['area', 'bedrooms', 'bathrooms', 'parking_spaces', 'latitude', 'longitude']
    features_cat = ['neighborhood']

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