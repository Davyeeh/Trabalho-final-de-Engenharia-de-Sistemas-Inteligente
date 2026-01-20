import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Configuração das Colunas (Baseado no Notebook) ---
# Colunas finais que o modelo usará (já renomeadas para inglês/padrão)
COLUNAS_NUMERICAS = ['area', 'bedrooms', 'bathrooms', 'parking_spaces', 'latitude', 'longitude', 'price']
COLUNAS_CATEGORICAS = ['neighborhood'] 
TARGET = 'price'

def _separar_endereco(texto):
    """
    Replica a lógica exata do notebook para separar Rua e Bairro.
    Fonte: Célula 35 do projeto_final_ESI.ipynb
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return pd.Series(["Não Informado", "Não Informado"])

    # Separa a Cidade/Estado (tudo depois do " - ")
    # No notebook você descarta a cidade depois, então vamos focar na parte1
    try:
        parte1, _ = texto.split(" - ", 1)
    except ValueError:
        parte1 = texto

    # Separa Rua e Bairro (tudo depois da última vírgula é o bairro)
    if "," in parte1:
        # rsplit faz a quebra da direita para a esquerda
        rua, bairro = parte1.rsplit(",", 1)
    else:
        rua = "Não Informado"
        bairro = parte1

    return pd.Series([rua.strip(), bairro.strip()])

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de limpeza fiel ao notebook projeto_final_ESI.ipynb.
    """
    df = df.copy()

    # 1. Tratamento de Nulos na Localização
    # Remover linhas onde Latitude OU Longitude estão vazias
    cols_loc = ['Latitude', 'Longitude']
    if all(c in df.columns for c in cols_loc):
        df = df.dropna(subset=cols_loc)

    # 2. Ajuste do Preço (Lógica IQR)
    if 'Price' in df.columns:
        # Garante numérico
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Filtra preços válidos para cálculo
        precos_validos = df.loc[df['Price'] > 0, 'Price']
        
        if not precos_validos.empty:
            Q1 = precos_validos.quantile(0.25)
            Q3 = precos_validos.quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 1.5 * IQR
            
            # Média justa sem outliers
            media_justa = precos_validos[precos_validos <= limite_superior].mean()
            
            # Substituição (evitando warnings de dtype)
            # Substitui 0 e NaN pela média
            mask_zero = (df['Price'] == 0) | (df['Price'].isna())
            df.loc[mask_zero, 'Price'] = float(media_justa)

    # 3. Remoção de Duplicatas
    df = df.drop_duplicates()

    # 4. Ajuste no Endereço
    if 'Adress' in df.columns:
        # Cria Rua e Bairro
        df[['street', 'neighborhood']] = df['Adress'].apply(_separar_endereco)
        # Remove coluna original (Cidade também é removida pois variância é 0)
        df = df.drop(columns=['Adress'], errors='ignore')

    # 5. Ajuste de Data
    if 'created_date' in df.columns:
        df['created_date'] = df['created_date'].astype(str).str.split('T').str[0]

    # 6. Remoção de colunas inúteis
    # Removemos extract_date também se não for usar, ou mantemos para log
    colunas_remover = ['ID', 'below_price', 'extract_date'] 
    df = df.drop(columns=colunas_remover, errors='ignore')

    # 7. RENOMEAÇÃO FINAL
    # Mapeia os nomes originais (CamelCase) para o padrão do sistema (snake_case)
    mapa_nomes = {
        'Area': 'area',
        'Bedrooms': 'bedrooms',       # No notebook é Bedrooms, não Rooms
        'Bathrooms': 'bathrooms',     # No notebook é Bathrooms, não Toilets
        'Parking_Spaces': 'parking_spaces', # No notebook é Parking_Spaces
        'Price': 'price',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        # street e neighborhood já criamos minúsculos
    }
    df = df.rename(columns=mapa_nomes)

    return df

def preparar_dados_para_treino(df_novo, caminho_historico):
    """
    Função principal de orquestração.
    """
    # 1. Limpeza
    df_limpo = limpar_dados(df_novo)

    # 2. Persistência do Histórico
    try:
        df_historico = pd.read_csv(caminho_historico)
        df_full = pd.concat([df_historico, df_limpo], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_full = df_limpo
    
    df_full.to_csv(caminho_historico, index=False)
    
    # 3. Seleção de Features
    cols_existentes = [c for c in COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS if c in df_full.columns]
    
    if TARGET not in df_full.columns:
         # Fallback: se não tiver price, tenta achar Price (caso a renomeação falhe)
         raise ValueError("Coluna alvo 'price' não encontrada.")

    X = df_full[cols_existentes]
    y = df_full[TARGET]
    
    return X, y

def criar_pre_processador():
    """
    Pipeline do Scikit-Learn atualizado com os novos nomes de colunas.
    """
    # Define as colunas numéricas (exceto o target)
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