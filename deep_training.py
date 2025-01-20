import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocess import preprocessing_pipeline
import joblib

# Carregar os dados
df = pd.read_csv('data/Abandono_clientes.csv')

df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
#df['Geo_Gender'] = df['Geography'].astype('str') + '_' + df['Gender'].astype('str') # Não foi relevante

X = df.drop('Exited', axis=1)
y = df['Exited']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# Criar uma única instância do pré-processador
preprocessor = preprocessing_pipeline()

# Criar o pipeline com pré-processamento, classificador
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, scale_pos_weight=2))
])

# Definir o grid de parâmetros ajustados para LGBMClassifier
param_grid = {
    'classifier__num_leaves': [11, 50],
    'classifier__learning_rate': [0.01],
    'classifier__n_estimators': [100, 1000],
}

# Inicializar o GridSearchCV com o pipeline e o grid de parâmetros
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Ajustar o modelo usando o GridSearchCV
grid_search.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = grid_search.predict(X_test)

# Avaliar o desempenho do modelo
print("\nMelhores Parâmetros Encontrados:")
print(grid_search.best_params_)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Definir X e y a partir de toda a base
X_full = df.drop('Exited', axis=1)
y_full = df['Exited']

# Remover o prefixo 'classifier__' dos melhores parâmetros
best_params = {k.split('__')[1]: v for k, v in grid_search.best_params_.items()}

# Criar um novo pipeline com os melhores parâmetros
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=42,
        scale_pos_weight=2,
        **best_params  # Usa os melhores parâmetros encontrados
    ))
])

# Ajustar o pipeline na base completa
final_pipeline.fit(X_full, y_full)

# Salvar o modelo treinado na base completa
joblib.dump(final_pipeline, 'modelo_final_treinado.joblib')

print("Modelo final treinado na base inteira e salvo com sucesso.")