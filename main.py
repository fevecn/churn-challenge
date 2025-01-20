import os
import subprocess
import pandas as pd
import joblib

# Carregar o modelo treinado
modelo_caminho = 'modelo_final_treinado.joblib'

# Verificar se o arquivo do modelo existe
if os.path.exists(modelo_caminho):
    # Carregar o modelo treinado
    modelo = joblib.load(modelo_caminho)
    print("Modelo carregado com sucesso:")
    print(modelo)
else:
    print(f"Modelo '{modelo_caminho}' não encontrado. Treinando o modelo...")

    # Rodar o script deep_training.py
    subprocess.run(['python', 'deep_training.py'], check=True)

    # Verificar novamente se o modelo foi gerado após o treinamento
    if os.path.exists(modelo_caminho):
        modelo = joblib.load(modelo_caminho)
        print("Modelo treinado e carregado com sucesso:")
        print(modelo)
    else:
        raise FileNotFoundError("O modelo não foi criado mesmo após o treinamento. Verifique 'deep_training.py'.")

# Carregar a nova base
df_teste = pd.read_csv('data/Abandono_teste.csv')

# Separar as colunas do arquivo delimitado por ponto e vírgula
df_teste = df_teste['RowNumber;CustomerId;Surname;CreditScore;Geography;Gender;Age;Tenure;Balance;NumOfProducts;HasCrCard;IsActiveMember;EstimatedSalary'].str.split(';', expand=True)

# Definir os nomes das colunas para a nova base
df_teste.columns = [
    "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# Converter as colunas numéricas para o tipo correto
num_columns = ["RowNumber", "CustomerId", "CreditScore", "Age", "Tenure", "Balance",
               "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

df_teste[num_columns] = df_teste[num_columns].apply(pd.to_numeric, errors='coerce')

# Criar a nova feature utilizada pelo modelo
df_teste['IsActive_by_CreditCard'] = df_teste['HasCrCard'] * df_teste['IsActiveMember']

# Verificar se a coluna 'RowNumber' está presente
if 'RowNumber' not in df_teste.columns:
    raise ValueError("A nova base deve conter a coluna 'RowNumber'.")

# Selecionar apenas as features que o modelo utiliza (excluindo colunas desnecessárias)
X = df_teste.drop(columns=['Exited'], errors='ignore')

# Fazer previsões
df_teste['predictedValues'] = modelo.predict(X)

# Criar a tabela final com 'RowNumber' e os valores previstos
output = df_teste[['RowNumber', 'predictedValues']]

# Salvar o arquivo CSV com os resultados
output_caminho = 'resultado_final.xlsx'
output.to_excel(output_caminho, index=False)

print(f"Arquivo com previsões salvo em: {output_caminho}")
