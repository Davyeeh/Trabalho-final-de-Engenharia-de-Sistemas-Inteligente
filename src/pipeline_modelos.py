import pandas as pd
import numpy as np
import logging
import sys
import pickle
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics
from sklearn.pipeline import Pipeline

def load_dataset(dataset_path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)

def extract_model_metrics_scores(y_test, y_pred) -> dict: 
    scores = {
        "r2_score": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    }
    return scores

def run_experiment(dataset, x_features, y_label, models, grid_params_list, cv_criteria) -> dict:
    X = dataset[x_features]
    y = dataset[y_label]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models_info_per_fold = {}
    
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        models_info = {}
        for model_name in models:
            grid_model = GridSearchCV(models[model_name], grid_params_list[model_name], cv=3, scoring=cv_criteria)                        
            grid_model.fit(X_train, y_train)
            
            y_pred = grid_model.predict(X_test)
            metrics_scores = extract_model_metrics_scores(y_test, y_pred)
            
            models_info[model_name] = {
                "score": metrics_scores,
                "best_estimator": grid_model.best_estimator_,
                "best_params": grid_model.best_params_
            }            
        models_info_per_fold[i] = models_info
        print(f"Fold {i} concluído.")

    return models_info_per_fold

def do_benchmark(x_features, y_label, grid_search=False, dataset_path=None, cv_criteria="r2", selected_models=["XGB", "RFR"]) -> dict:    
    dataset = load_dataset(dataset_path)
    
    train_models = {
        "LR": LinearRegression(),
        "RFR": RandomForestRegressor(random_state=42),
        "XGB": XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    models = {i: train_models[i] for i in train_models if i in selected_models}
              
    if grid_search:            
        grid_params_list = {
            "LR": { 
                "fit_intercept": [True, False] 
            },
            "RFR": { 
                "n_estimators": [50, 100],        
                "max_depth": [10, 20, None],      
                "min_samples_split": [2, 5],      
                "max_features": ["sqrt", 1.0]     
            },
            "XGB": { 
                "n_estimators": [100, 150],       
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],           
                "subsample": [0.8, 1.0],          
                "colsample_bytree": [0.8, 1.0]    
            }    
        }
    else:
        grid_params_list = {"LR": {}, "RFR": {}, "XGB": {}}
    
    print(f"Iniciando Benchmark com features: {x_features}")
    
    fold_results = run_experiment(dataset=dataset, x_features=x_features, 
                                  y_label=y_label, 
                                  models=models, grid_params_list=grid_params_list, 
                                  cv_criteria=cv_criteria)
    return fold_results

def select_best_model(fold_results):
    results = {}
    for fold in fold_results.keys():
        for model_name in fold_results.get(fold).keys():
            r2 = fold_results[fold][model_name]["score"]["r2_score"]
            if results.get(model_name) is None:
                results[model_name] =  { "r2": []}
            results[model_name]["r2"].append(r2)

    best_model_score = -np.inf
    best_model_name = None
    for key in results.keys():
        avg_r2 = np.mean(results[key]["r2"])
        if avg_r2 > best_model_score:
            best_model_score = avg_r2
            best_model_name = key
    return best_model_name

# --- FUNÇÃO PRINCIPAL USADA PELO APP.PY ---
def treinar_e_salvar(X, y, preprocessor, caminho_modelo):
    """Treina o modelo final usando Pipeline completo."""
    
    modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    pipeline_completo = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', modelo)
    ])
    
    print(f"Iniciando treinamento do modelo campeão com {len(X)} amostras...")
    pipeline_completo.fit(X, y)
    
    score = pipeline_completo.score(X, y) # R2
    print(f"Modelo treinado com R²: {score:.4f}")
    
    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)
    joblib.dump(pipeline_completo, caminho_modelo)
    
    return score

def start(dataset_path):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='pipeline.log', encoding='utf-8', level=logging.DEBUG)
    
    # --- AJUSTADO PARA COLUNAS MAIÚSCULAS ---
    features_numericas_teste = ['Area', 'Bedrooms', 'Bathrooms', 'Parking_Spaces', 'Latitude', 'Longitude', 'Created_Date']
    
    logger.debug("[Step-1] Benchmark Rápido")
    try:
        fold_results = do_benchmark(
            x_features=features_numericas_teste,
            y_label="Price", # Agora com P maiúsculo
            grid_search=True, 
            dataset_path=dataset_path, 
            selected_models=["XGB", "RFR"]
        )
        best_model = select_best_model(fold_results)
        print(f"Melhor modelo no benchmark: {best_model}")
        
    except Exception as e:
        print(f"Erro no benchmark: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        start(str(sys.argv[1]))
    else:
        padrao = "dados/historico_apartamentos.csv"
        if os.path.exists(padrao):
            start(padrao)
        else:
            print(f"Arquivo não encontrado: {padrao}")