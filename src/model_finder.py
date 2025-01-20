# model_finder.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump
from sklearn.pipeline import Pipeline

# Importa a função que cria o pipeline de pré-processamento
from src.preprocess import preprocessing_pipeline

def model_finder(df):
    """
    Recebe um DataFrame completo, com colunas que incluem 'Exited' como alvo.
    Roda GridSearch para descobrir o melhor modelo e hiperparâmetros.
    Salva as infos em 'melhor_modelo_info.joblib'.
    Retorna (info) = {'modelo_escolhido', 'best_params', 'best_score'}.
    """
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']

    y = df['Exited']
    X = df.drop(columns=['Exited'])
    print('Iniciando modelo...')

    # 1) Chama a pipeline base
    #    - "preprocessing" vem do módulo 'preprocessing.py'
    #    - "classifier" será o placeholder para vários modelos
    preprocessing = preprocessing_pipeline()
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', LogisticRegression(random_state=42))  # Placeholder
    ])
    print('Chamando pipeline de preprocessing...')


    # Dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # 2) Define param_grid: haverá a troca de 'classifier' e seus parâmetros
    param_grid = [
        # Logistic Regression
        {
            'classifier': [LogisticRegression(solver='liblinear', random_state=42)],
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        },
        # Random Forest
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        # XGBoost
        {
            'classifier': [XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)],
            'classifier__n_estimators': [50, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.7, 1]
        },
        # LightGBM
        {
            'classifier': [LGBMClassifier(random_state=42, scale_pos_weight=2)],
            'classifier__n_estimators': [100, 1000],
            'classifier__num_leaves': [11, 50],
            'classifier__learning_rate': [0.01],
        },
    ]

    # 3) Rodar GridSearch
    print('Rodando GridSearch...')
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # 7. Avaliar o desempenho do modelo
    y_pred = best_estimator.predict(X_test)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # 4) Extrair nome do modelo
    modelo_escolhido = best_estimator.named_steps['classifier'].__class__.__name__
    print('Modelo escolhido:', modelo_escolhido)

    # 5) Salvar as infos
    info = {
        'modelo_escolhido': modelo_escolhido,
        'best_params': best_params,
        'best_score': best_score,
        'best_estimator': best_estimator
    }
    dump(info, '../melhor_modelo_info.joblib')
    print('Info salvo com sucesso!')
    return info

if __name__ == '__main__':
    df = pd.read_csv("../data/Abandono_clientes.csv")
    info = model_finder(df)
    print(info)
