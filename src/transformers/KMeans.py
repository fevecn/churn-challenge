import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class KMeansClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, features, n_clusters=20, random_state=0, n_components=None):
        """
        features: lista com nomes das colunas numéricas que serão usadas para o clustering
        n_clusters: número de clusters do KMeans
        random_state: semente para reprodutibilidade
        n_components: se != None, aplica PCA reduzindo para 'n_components'
        """
        self.features = features
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_components = n_components

        # Aqui instanciamos o KMeans, StandardScaler e PCA
        # mas só faremos fit deles nos métodos fit/transform.
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=50,
            random_state=self.random_state
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        # X deverá ser um DataFrame contendo pelo menos as colunas de self.features
        X_scaled = self.scaler.fit_transform(X[self.features])

        if self.n_components is not None:
            X_scaled = self.pca.fit_transform(X_scaled)

        # Ajusta o KMeans
        self.kmeans.fit(X_scaled)
        return self

    def transform(self, X):
        # Transforma (escala e aplica PCA, se houver) as colunas
        X_scaled = self.scaler.transform(X[self.features])

        if self.n_components is not None:
            X_scaled = self.pca.transform(X_scaled)

        # Obtém rótulos de cluster
        cluster_labels = self.kmeans.predict(X_scaled)

        # Cria DataFrame com a nova coluna "Cluster"
        X_new_cluster = pd.DataFrame({"Cluster": cluster_labels})

        # Concatena o DataFrame original (ou cópia) com a coluna de cluster
        X_copy = X.copy().reset_index(drop=True)
        X_new_cluster = X_new_cluster.reset_index(drop=True)

        return pd.concat([X_copy, X_new_cluster], axis=1)
