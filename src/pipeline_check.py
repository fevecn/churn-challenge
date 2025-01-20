# pipeline_check.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocess import preprocessing_pipeline
from imblearn.base import BaseSampler  # Importar BaseSampler para identificar resamplers

def load_data(filepath):
    """
    Carrega os dados a partir de um arquivo CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo {filepath} não foi encontrado.")
    df = pd.read_csv(filepath)
    return df

def check_values(df, step_name):
    """
    Verifica a presença de NaNs, Infs, -Infs e valores muito grandes em um DataFrame.
    Imprime a quantidade de ocorrências por coluna.
    """
    print(f"\n[Verificação] Etapa: {step_name}")

    # Verificação de NaNs
    total_nans = df.isna().sum().sum()
    if total_nans > 0:
        print(f"  - Total de NaNs: {total_nans}")
        print("  - Quantidade de NaNs por coluna:")
        print(df.isna().sum())
    else:
        print("  - Nenhum NaN encontrado.")

    # Seleciona apenas colunas numéricas para verificar Infs e valores muito grandes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # Verificação de Infs
    total_infs = np.isinf(df_numeric).sum().sum()
    if total_infs > 0:
        print(f"  - Total de Infs: {total_infs}")
        print("  - Quantidade de Infs por coluna:")
        print(np.isinf(df_numeric).sum())
    else:
        print("  - Nenhum Inf encontrado.")

    # Verificação de valores muito grandes (>1e10)
    total_large = (df_numeric.abs() > 1e10).sum().sum()
    if total_large > 0:
        print(f"  - Total de valores > 1e10: {total_large}")
        print("  - Quantidade de valores > 1e10 por coluna:")
        print((df_numeric.abs() > 1e10).sum())
    else:
        print("  - Nenhum valor > 1e10 encontrado.")

def main():
    # Caminho para os dados
    data_path = "../data/Abandono_clientes.csv"

    # Carrega os dados
    print("Carregando os dados...")
    df = load_data(data_path)

    # Define X e y
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Divisão em treino e teste
    print("\nDividindo os dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # Verifica valores no conjunto de treino original
    print("\nVerificando valores no conjunto de treino original...")
    check_values(X_train, "Conjunto de Treino Original")

    # Obtém as etapas de pré-processamento
    preprocessing_steps = preprocessing_pipeline()

    # Inicializa um DataFrame temporário para aplicar as transformações
    temp_X = X_train.copy()
    temp_y = y_train.copy()

    print("\nIniciando a aplicação das etapas de pré-processamento...")

    for step_name, transformer in preprocessing_steps:
        print(f"\nAplicando a etapa: '{step_name}'")
        try:
            # Verifica se o transformer é um resampler
            if isinstance(transformer, BaseSampler):
                # Aplicar fit_resample para resamplers como SMOTE
                temp_X, temp_y = transformer.fit_resample(temp_X, temp_y)
                print(f"  - Resampling realizado. Novo tamanho de X: {temp_X.shape}, y: {temp_y.shape}")
            else:
                # Aplicar fit_transform ou transform para transformadores
                if hasattr(transformer, 'fit_transform'):
                    temp_X = transformer.fit_transform(temp_X, temp_y)
                else:
                    temp_X = transformer.transform(temp_X)

                # Se a transformação retornar um DataFrame, mantenha-o; caso contrário, converta para DataFrame
                if not isinstance(temp_X, pd.DataFrame):
                    # Tenta obter nomes de features se disponíveis
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names = transformer.get_feature_names_out()
                    elif hasattr(transformer, 'get_feature_names'):
                        feature_names = transformer.get_feature_names()
                    else:
                        feature_names = None
                    temp_X = pd.DataFrame(temp_X, columns=feature_names)

                # Verifica valores após a transformação
                check_values(temp_X, step_name)

        except Exception as e:
            print(f"\n[ERRO] Erro ao aplicar a etapa '{step_name}': {e}")
            break

    print("\nDiagnóstico concluído.")

if __name__ == "__main__":
    main()
