import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class VariableBinning(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, column_name=None, strategy='uniform'):
        """
        Inicializa o transformador de binagem.

        Parameters:
        - n_bins (int): Número de bins a serem criados.
        - column_name (str): Nome da coluna a ser binada.
        - strategy (str): Estratégia de binagem ('uniform', 'quantile').
        """
        self.n_bins = n_bins
        self.column_name = column_name
        self.strategy = strategy
        self.bins = None

    def fit(self, X, y=None):
        """
        Calcula os limites dos bins com base na coluna especificada.

        Parameters:
        - X (pd.DataFrame): DataFrame de entrada.
        - y: Não utilizado, apenas para compatibilidade.

        Returns:
        - self
        """
        if self.column_name is None:
            raise ValueError("O parâmetro 'column_name' deve ser especificado.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("A entrada X deve ser um pandas DataFrame.")

        if self.strategy == 'uniform':
            min_val = X[self.column_name].min()
            max_val = X[self.column_name].max()
            # Adiciona uma pequena margem ao max_val para garantir inclusão
            epsilon = 1e-8
            self.bins = np.linspace(min_val, max_val + epsilon, self.n_bins + 1)
        elif self.strategy == 'quantile':
            self.bins = X[self.column_name].quantile(
                np.linspace(0, 1, self.n_bins + 1)
            ).unique()
            # Garantir que há n_bins + 1 limites únicos
            if len(self.bins) < self.n_bins + 1:
                raise ValueError("Não há bins suficientes para a estratégia 'quantile'. Considere reduzir o número de bins.")
        else:
            raise ValueError("A estratégia de binagem deve ser 'uniform' ou 'quantile'.")

        return self

    def transform(self, X):
        """
        Aplica a binagem à coluna especificada e adiciona a nova coluna ao DataFrame.

        Parameters:
        - X (pd.DataFrame): DataFrame de entrada.

        Returns:
        - X_transformed (pd.DataFrame): DataFrame com a nova coluna de binagem.
        """
        if self.column_name is None:
            raise ValueError("O parâmetro 'column_name' deve ser especificado.")

        if self.bins is None:
            raise AttributeError("O método fit deve ser chamado antes de transform.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("A entrada X deve ser um pandas DataFrame.")

        X = X.copy()

        bin_column = f"{self.column_name}_Bin"
        # Utiliza labels numéricos começando em 0
        X[bin_column] = pd.cut(
            X[self.column_name],
            bins=self.bins,
            labels=False,
            include_lowest=True,
            right=True
        )

        # Verifica e trata valores fora dos bins
        # Substitui NaNs resultantes de valores fora dos bins por um bin específico (e.g., o último bin)
        if X[bin_column].isna().sum() > 0:
            print(f"Encontrados {X[bin_column].isna().sum()} valores fora dos intervalos após a binagem.")
            # Opcional: Atribui esses valores ao último bin
            X[bin_column] = X[bin_column].fillna(self.n_bins - 1)

        return X
