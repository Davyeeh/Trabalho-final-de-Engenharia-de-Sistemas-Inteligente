# 🏢 Predição de Preços de Apartamentos em São Paulo (API)

Este projeto é o **Trabalho Final de Engenharia de Sistemas Inteligentes**. Ele consiste em uma solução completa de **Machine Learning** para prever preços de venda de apartamentos na cidade de São Paulo, utilizando dados como área, localização (bairro), número de quartos e vagas de garagem.

A solução expõe uma **API REST** construída com **FastAPI**, permitindo o treinamento do modelo e a realização de predições de forma programática.

---

## 🚀 Como Executar o Projeto

Siga os passos abaixo para rodar a aplicação localmente.

### Pré-requisitos

- Python 3.12+  
- Poetry (Gerenciador de dependências)

### Passo a Passo

1. Clone o repositório:

```bash
git clone https://github.com/davyeeh/trabalho-final-de-engenharia-de-sistemas-inteligente.git
cd trabalho-final-de-engenharia-de-sistemas-inteligente
```

2. Instale as dependências:

```bash
poetry install
```

3. Ative o ambiente virtual:

```bash
poetry shell
```

4. Inicie a API:

```bash
python app.py
# Ou, se preferir usar o uvicorn diretamente:
# uvicorn app:app --reload
```

A API estará rodando em:  
👉 http://127.0.0.1:8000

Acesse a documentação (Swagger UI):  
👉 http://127.0.0.1:8000/docs

---

## 🛠️ Estrutura do Projeto

```plaintext
├── app.py                   # Ponto de entrada da API (FastAPI) e rotas
├── pyproject.toml           # Configuração do projeto e dependências (Poetry)
├── README.md                # Documentação do projeto
│
├── artifacts/               # Pasta onde o modelo treinado (.pkl) é salvo
│   └── modelo_campeao.pkl
│
├── dados/                   # Armazenamento do histórico de dados processados
│   └── historico_apartamentos.csv
│
├── dataset/                 # Dados brutos originais (fonte de dados)
│   ├── dataset_original.csv
│   └── SaoPaulo_OnlyAppartments_2024-11-25.csv
│
├── notebooks/               # Jupyter Notebooks para análises e testes
│   ├── projeto_final_ESI.ipynb
│   └── Treinamento_ESI.ipynb
│
└── src/                     # Código fonte do pipeline de ML
    ├── pipeline_dados.py    # Lógica de limpeza e engenharia de features
    └── pipeline_modelos.py  # Lógica de treinamento e validação
```

---

## 🧠 Funcionamento do Pipeline

### 1. Tratamento de Dados (`src/pipeline_dados.py`)

O sistema recebe os dados brutos e aplica as seguintes transformações:

- **Limpeza:** Remoção de nulos em coordenadas (Latitude/Longitude).
- **Outliers:** Filtragem de preços discrepantes usando IQR (Interquartile Range).
- **Engenharia de Features:**
  - Extração do ano da data de criação do anúncio.
  - Padronização de nomes de bairros.
- **Padronização:** Renomeia colunas para o padrão esperado pelo modelo (CamelCase).

### 2. Modelagem (`src/pipeline_modelos.py`)

Pipeline do Scikit-Learn contendo:

- **Pré-processador (ColumnTransformer):**
  - Variáveis Numéricas: imputação pela mediana + `StandardScaler`.
  - Variáveis Categóricas (bairro): `OneHotEncoder`.
- **Modelo Preditivo:**
  - XGBoost Regressor ou Random Forest Regressor.
  - Otimização baseada em RMSE e R².

---

## 📡 Endpoints da API

### `POST /train`

Retreina o modelo a partir de um arquivo CSV ou Excel.

- **Input:** `.csv` ou `.xlsx`
- **Ação:**  
  - Limpa os dados  
  - Atualiza o histórico (`dados/`)  
  - Treina o modelo  
  - Salva em `artifacts/`

### `POST /predict`

Retorna o preço estimado de um imóvel.

**Body (JSON):**

```json
{
  "area": 57,
  "bedrooms": 2,
  "bathrooms": 1,
  "parking_spaces": 1,
  "neighborhood": "Pinheiros",
  "latitude": -23.56,
  "longitude": -46.69,
  "created_date": 2024
}
```

**Response:**

```json
{
  "preco_estimado": 650000.0,
  "neighborhood": "Pinheiros"
}
```

---

## 🧪 Testes e Notebooks

Para testar o fluxo completo sem Postman ou Insomnia, utilize:

- `notebooks/Treinamento_ESI.ipynb`
- `projeto_final_ESI.ipynb`

Esses notebooks permitem:

- Carregar o dataset original  
- Enviar dados para a rota de treino  
- Executar predições de teste  
