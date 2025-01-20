import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SalaryBalanceRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 salary_col='EstimatedSalary',
                 balance_col='Balance',
                 new_column='EstimatedSalary_Balance_Ratio',
                 drop_original=False,
                 fill_value=0.000001):  # Mantido como 0 ou outro valor desejado
        """
        Inicializa o transformador para calcular a proporção de EstimatedSalary por Balance.

        Parameters:
        - salary_col (str): Nome da coluna de EstimatedSalary.
        - balance_col (str): Nome da coluna de Balance.
        - new_column (str): Nome da nova coluna resultante da proporção.
        - drop_original (bool): Se True, remove as colunas originais após a criação da nova coluna.
        - fill_value (float): Valor a ser atribuído onde Balance é zero para evitar divisão por zero.
        """
        self.salary_col = salary_col
        self.balance_col = balance_col
        self.new_column = new_column
        self.drop_original = drop_original
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """
        Ajusta o transformador. Neste caso, não há parâmetros a serem aprendidos.

        Parameters:
        - X (pd.DataFrame): DataFrame de entrada.
        - y: Não utilizado, apenas para compatibilidade.

        Returns:
        - self
        """
        # Verifica se as colunas existem no DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("A entrada X deve ser um pandas DataFrame.")

        for col in [self.salary_col, self.balance_col]:
            if col not in X.columns:
                raise ValueError(f"A coluna '{col}' não está presente no DataFrame.")

        return self

    def transform(self, X):
        """
        Aplica a transformação de cálculo da proporção de EstimatedSalary por Balance.

        Parameters:
        - X (pd.DataFrame): DataFrame de entrada.

        Returns:
        - X_transformed (pd.DataFrame): DataFrame com a nova coluna adicionada.
        """
        X = X.copy()

        # Identificar onde Balance é zero
        balance_zero = X[self.balance_col] == 0
        zeros_before = balance_zero.sum()

        if zeros_before > 0:
            print(f"Encontradas {zeros_before} ocorrências de '{self.balance_col}' iguais a zero. Atribuindo '{self.fill_value}' à nova coluna para esses casos.")

        # Calcular a proporção, evitando divisão por zero
        X[self.new_column] = X[self.salary_col] / X[self.balance_col]

        # Atribuir fill_value onde Balance era zero
        X.loc[balance_zero, self.new_column] = self.fill_value

        # Opcionalmente, remover as colunas originais
        if self.drop_original:
            X.drop([self.salary_col, self.balance_col], axis=1, inplace=True)

        return X
