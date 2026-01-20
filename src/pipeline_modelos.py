import pandas
import pickle
import logging
import sys

from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.pipeline import Pipeline

def load_dataset(dataset_path) -> pandas.DataFrame:
    """
    Carrega um arquivo CSV com os dados imobiliarios.
    """
    df_dataset = pandas.read_csv(dataset_path)

    return df_dataset


def save_model(model, model_name, data_balance, cv_criteria):
    """
    Salva o modelo fornecido em um arquivo pickle.

    Parameters:
        model (obj): Instância do modelo a ser salva.
        model_name (str): Nome identificador do modelo.
        data_balance (str): Método de balanceamento de dados utilizado.
        cv_criteria (str): Critério de validação cruzada utilizado.
    """      
    with open(f"model_{model_name}_{cv_criteria.upper()}_{data_balance}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

def load_model(file_model_path):
    """
    Carrega um modelo previamente salvo em um arquivo pickle.

    Parameters:
        file_model_path (str): Caminho para o arquivo .pkl contendo o modelo.

    Returns:
        model (obj): O modelo carregado.
    """    
    return pickle.load(open(file_model_path, 'rb'))

def extract_model_metrics_scores(y_test, y_pred)  -> dict: 
    """
    Extrai métricas de desempenho comparando predições e valores reais.

    Parameters:
        y_test (array-like): Rótulos reais do conjunto de teste.
        y_pred (array-like): Rótulos previstos pelo modelo.

    Returns:
        scores (dict): Dicionário com métricas como acurácia, ROC-AUC, F1, etc.
    """  
    scores = {"accuracy_score": metrics.accuracy_score(y_test, y_pred),
              "roc_auc_score": metrics.roc_auc_score(y_test, y_pred),
              "f1_score": metrics.f1_score(y_test, y_pred),
              "precision_score": metrics.precision_score(y_test, y_pred),
              "recall_score": metrics.recall_score(y_test, y_pred),
              "matthews_corrcoef": metrics.matthews_corrcoef(y_test, y_pred),
              "brier_score_loss": metrics.brier_score_loss(y_test, y_pred),
              "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
              "classification_report": metrics.classification_report(y_test, y_pred)}
    return scores


def run_experiment(dataset, x_features, y_label, data_balance, models, grid_params_list, cv_criteria) -> dict:
    """
    Executa benchmark de diversos modelos usando validação cruzada estratificada.

    Parameters:
        dataset (pandas.DataFrame): DataFrame contendo as features e rótulos.
        x_features (list of str): Lista com nomes das colunas das features.
        y_label (str): Nome da coluna do rótulo.
        data_balance (str): Método para balanceamento de dados ('SMOTE' ou outro).
        models (dict): Dicionário nome->instância dos modelos a avaliar.
        grid_params_list (dict): Dicionário nome->parâmetros do GridSearchCV.
        cv_criteria (str): Critério de scoring para o GridSearchCV (ex: 'roc_auc').

    Returns:
        fold_results (dict): Dicionário onde cada chave é o índice do fold e o valor
                             é outro dicionário com informações de cada modelo
                             ('score' e 'best_estimator').
    """    
    X = dataset[x_features]
    y = dataset[y_label]

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    models_info_per_fold = {}
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if data_balance=='SMOTE':
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        models_info = {}
        for model in models:
            grid_model = GridSearchCV(models[model], grid_params_list[model], cv=5, scoring=cv_criteria)                        
            grid_model.fit(X_train, y_train)
            y_pred = grid_model.predict(X_test)
            metrics_scores = extract_model_metrics_scores(y_test, y_pred)
            models_info[model] = {
                "score": metrics_scores,
                "best_estimator": grid_model.best_estimator_
            }            

        models_info_per_fold[i] = models_info

    return models_info_per_fold

def do_benchmark(x_features, y_label,grid_search=False, data_balance="SMOTE", dataset_path=None, cv_criteria="roc_auc", selected_models=["LRC", "RFC", "SVC"]) -> dict:    
    """
    Orquestra o benchmark de modelos selecionados a partir de um arquivo de dados.

    Parameters:
        x_features (list of str): Nomes das colunas das features.
        y_label (str): Nome da coluna do rótulo.
        grid_search (bool): Se True, utiliza GridSearchCV para otimização de hiperparâmetros.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        dataset_path (str): Caminho para o arquivo CSV do dataset.
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').
        selected_models (list of str): Lista de chaves dos modelos a serem treinados.

    Returns:
        fold_results (dict): Resultado do benchmark para cada fold.
    """    

    dataset =  load_dataset(dataset_path)
    
    train_models = {"SVC": LinearSVC(),
                    "LRC": LogisticRegression(),
                    "RFC": RandomForestClassifier()}

    models = {i:train_models[i] for i in train_models if i in selected_models}
              
    if grid_search:            
        grid_params_list = {
                            "LRC":{"C":[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                                   "penalty":["l1","l2"],
                                   "max_iter":[10**7],
                                   "fit_intercept":[True],
                                   "solver":["liblinear"]},

                            "RFC":{"n_estimators":[5,30,50,75, 100, 150, 200],
                                   "max_depth": [4,5,6,7,8,None]},
            
                            "SVC":{"C":[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                                   "max_iter":[10**7]}                             
                            }
    else:
        grid_params_list = {"LRC":{}, "RFC":{}, "SVC":{}}
    
    fold_results = run_experiment(dataset=dataset, x_features=x_features, 
                                  y_label=y_label, data_balance=data_balance, 
                                  models=models, grid_params_list=grid_params_list, 
                                  cv_criteria=cv_criteria)

    return fold_results

def build_champion_model(dataset, x_features, y_label, data_balance, model_info, cv_criteria) -> dict:
    """
    Cria o modelo final (champion model) a partir de divisão treino/teste e balanceamento.

    Parameters:
        dataset (pandas.DataFrame): DataFrame com features e rótulos completos.
        x_features (list of str): Nomes das colunas das features.
        y_label (str): Nome da coluna do rótulo.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        model_info (dict): Informações sobre o modelo ('instance', 'grid_params_list', 'name').
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').

    Returns:
        metrics_scores (dict): Métricas de desempenho do modelo final.
    """    
    X = dataset[x_features]
    y = dataset[y_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    if data_balance=='SMOTE':
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)        
    
    grid_model = GridSearchCV(model_info.get("instance"), model_info.get("grid_params_list"), cv=5, scoring=cv_criteria)                        
    grid_model.fit(X_train, y_train)
    y_pred = grid_model.predict(X_test)
    metrics_scores = extract_model_metrics_scores(y_test, y_pred)

    save_model(grid_model.best_estimator_, model_info.get("name"), data_balance, cv_criteria)

    return metrics_scores


def make_model(x_features, y_label, grid_search=False, data_balance="SMOTE", dataset_path=None, cv_criteria="roc_auc", selected_model=None) -> dict:    
    """
    Constrói e avalia o modelo selecionado como champion.

    Parameters:
        x_features (list of str): Nomes das colunas das features.
        y_label (str): Nome da coluna do rótulo.
        grid_search (bool): Se True, usa GridSearchCV para otimizar hiperparâmetros.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        dataset_path (str): Caminho para o arquivo CSV do dataset.
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').
        selected_model (str): Chave do modelo a treinar (e.g., 'LRC', 'RFC', 'SVC').

    Returns:
        metrics_scores (dict): Métricas de desempenho do modelo final.
    """    

    dataset =  load_dataset(dataset_path)
    
    train_models = {"SVC": LinearSVC(),
                    "LRC": LogisticRegression(),
                    "RFC": RandomForestClassifier()}

    if grid_search:
        grid_params_list = {
                            "LRC":{"C":[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                                   "penalty":["l1","l2"],
                                   "max_iter":[10**7],
                                   "fit_intercept":[True],
                                   "solver":["liblinear"]},

                            "RFC":{"n_estimators":[5,30,50,75, 100, 150, 200],
                                   "max_depth": [4,5,6,7,8,None]},
            
                            "SVC":{"C":[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                                   "max_iter":[10**7]}                             
                            }
    else:
        grid_params_list = {"LRC":{}, "RFC":{}, "SVC":{}}
    
    model_info = {
        "name": selected_model,
        "instance": train_models.get(selected_model),
        "grid_params_list":grid_params_list.get(selected_model)
    }    

    metrics_scores = build_champion_model(dataset, x_features, y_label, data_balance, 
                                          model_info, cv_criteria)
    return metrics_scores

def select_best_model(fold_results, selected_models=["LRC", "RFC", "SVC"]):
    """
    Seleciona o modelo com maior média de ROC-AUC a partir dos resultados por fold.

    Parameters:
        fold_results (dict): Resultado do benchmark, dicionário de folds->modelos.
        selected_models (list of str): Modelos considerados (não usado diretamente no cálculo).

    Returns:
        best_model_name (str): Nome do modelo com maior média de ROC-AUC.
    """    
    results = {}
    for fold in fold_results.keys():
        for model_name in fold_results.get(fold).keys():
            roc_auc = fold_results[fold][model_name]["score"]["roc_auc_score"]
            if results.get(model_name) is None:
                results[model_name] =  { "roc_auc": []}
                results[model_name]["roc_auc"].append(roc_auc)
            else:
                results[model_name]["roc_auc"].append(roc_auc)

    best_model_score = 0
    best_model_name = None
    for  key in results.keys():
        avg_roc_auc = sum(results[key]["roc_auc"])/len(results[key]["roc_auc"])
        if avg_roc_auc > best_model_score:
            best_model_score = avg_roc_auc
            best_model_name = key
    
    return best_model_name
    
def start(dataset_path):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='pipeline.log', encoding='utf-8', level=logging.DEBUG)
    logger.debug("[Step-1] Realizando Benchmark")
    fold_results = do_benchmark(
                                x_features=['created_date', 'Area', 'Bedrooms', 'Bathrooms', 'Parking_Spaces', 'extract_date', 'Latitude', 'Longitude', 'Rua', 'Bairro'],
                                y_label="Price",
                                grid_search=True, 
                                dataset_path=dataset_path, 
                                selected_models=["LRC", "RFC"])
    for fold, models_info in fold_results.items():
            for model_name, info in models_info.items():
                sc = info['score']
                logger.debug(
                    f"Fold {fold} - Model {model_name}: "
                    f"ROC-AUC={sc['roc_auc_score']}, F1={sc['f1_score']}, "
                    f"Acc={sc['accuracy_score']}, Pre={sc['precision_score']}, Rec={sc['recall_score']}"
                )
    
    logger.debug("\n[Step-2] Selecionando Melhor Modelo")
    best_model_name = select_best_model(fold_results)
    logger.debug(f"\nBest Model {best_model_name}")
    
    logger.debug("\n[Step-3] Criando o Modelo Final")
    metric_scores = make_model(grid_search=True, 
                               dataset_path=dataset_path, 
                               selected_model=best_model_name)
    sc = metric_scores
    logger.debug(
        f"Champion {best_model_name}: ROC-AUC={sc['roc_auc_score']}, "
        f"F1={sc['f1_score']}, Acc={sc['accuracy_score']}, "
        f"Pre={sc['precision_score']}, Rec={sc['recall_score']}"
    )

    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = str(sys.argv[1])
        start(dataset_path)
    else:
        print("Você deve prover o caminho para o dataset.")    
