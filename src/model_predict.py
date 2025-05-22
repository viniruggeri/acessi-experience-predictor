import pickle
import pandas as pd
import numpy as np

def carregar_modelo(caminho='models/melhor_modelo.pkl'):
    with open(caminho, 'rb') as f:
        modelo, preprocessor = pickle.load(f)
    return modelo, preprocessor


def prever_nota(dados_dict, modelo, preprocessor):
    feature_cols = [
        'Faixa etária', 'Frequência de uso', 'Necessidade especial',
        'Horários de uso', 'Problemas enfrentados', 'Reação a problemas',
        'Planejamento de viagem', 'Tempo de viagem',
        'Perda de compromissos', 'Uso de apps', 'Funcionalidades preferidas',
        'Como gostaria de receber alertas', 'Uso de assistente virtual'
    ]


    df = pd.DataFrame([dados_dict])

    # Garante que todas as colunas existam (preenchendo com NaN)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Reordenar colunas
    df = df[feature_cols]

    # Aplicar pré-processamento e prever
    X_proc = preprocessor.transform(df)
    nota = modelo.predict(X_proc)[0]
    return round(float(nota), 2)