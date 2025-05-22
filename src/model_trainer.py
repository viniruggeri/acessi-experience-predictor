import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_prep import carregar_e_preprocessar_dados


def treinar_modelos(caminho_csv, salvar_em='models/melhor_modelo.pkl'):
    X, y, preprocessor = carregar_e_preprocessar_dados(caminho_csv)

    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    modelos = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    resultados = {}

    
    for nome, model in modelos.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_val_proc)
        resultados[nome] = {
            'RMSE': mean_squared_error(y_val, y_pred, squared=False),
            'MAE': mean_absolute_error(y_val, y_pred),
            'R2': r2_score(y_val, y_pred)
        }

    
    melhor_nome = min(resultados, key=lambda n: resultados[n]['RMSE'])
    melhor_modelo = modelos[melhor_nome]

    
    X_full_proc = preprocessor.fit_transform(pd.concat([X_train, X_val]))
    y_full = pd.concat([y_train, y_val])
    melhor_modelo.fit(X_full_proc, y_full)

    
    with open(salvar_em, 'wb') as f:
        pickle.dump((melhor_modelo, preprocessor), f)

    
    y_test_pred = melhor_modelo.predict(preprocessor.transform(X_test))
    resultados['Teste Final'] = {
        'RMSE': mean_squared_error(y_test, y_test_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred)
    }

    return resultados


if __name__ == "__main__":
    import pprint
    resultados = treinar_modelos('data/dados.csv')
    pprint.pprint(resultados)