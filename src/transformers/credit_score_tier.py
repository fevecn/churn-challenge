from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CreditScoreTierTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma a coluna 'CreditScore' em uma nova coluna 'ScoreTier' baseada em faixas numéricas:
        0-400      -> 1
        401-600    -> 2
        601-800    -> 3
        801-1000   -> 4
    """
    def __init__(self, score_column='CreditScore', new_column='ScoreTier'):
        self.score_column = score_column
        self.new_column = new_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Verifica se X é um DataFrame do pandas
        if isinstance(X, pd.DataFrame):
            X_ = X.copy()
            scores = X_[self.score_column]
        else:
            # Assume que X é um array numpy e que a coluna está na primeira posição
            X_ = pd.DataFrame(X, columns=[self.score_column])
            scores = X_[self.score_column]

        # Define os bins e os rótulos numéricos
        bins = [0, 400, 600, 800, 1000]
        labels = [1, 2, 3, 4]  # Números representando cada tier

        # Utiliza pd.cut para criar os tiers de forma vetorizada
        tiers = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)

        # Adiciona a nova coluna ao DataFrame
        X_[self.new_column] = tiers

        return X_