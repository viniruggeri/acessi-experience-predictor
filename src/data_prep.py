import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def carregar_e_preprocessar_dados(caminho_csv="data/dados.csv"):

    df = pd.read_csv(caminho_csv)


    feature_cols = [
        'Faixa etária', 'Frequência de uso', 'Necessidade especial',
        'Horários de uso', 'Problemas enfrentados', 'Reação a problemas',
        'Planejamento de viagem', 'Tempo de viagem',
        'Perda de compromissos', 'Uso de apps', 'Funcionalidades preferidas',
        'Como gostaria de receber alertas', 'Uso de assistente virtual'
    ]
    target_col = 'Nota da experiência'


    X = df[feature_cols]
    y = df[target_col]

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, feature_cols)
    ])

    return X, y, preprocessor