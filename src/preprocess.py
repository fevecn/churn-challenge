from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.transformers.KMeans import KMeansClusterer
from src.transformers.credit_score_tier import CreditScoreTierTransformer
from src.transformers.salary_balance_ratio import SalaryBalanceRatioTransformer
from src.transformers.variable_binner import VariableBinning
from src.transformers.drop_column import DropColumn

categorical_cols = ['Geography', 'Gender']

numerical_cols = ['CreditScore',
                  'Age',
                  'Tenure',
                  'Balance',
                  'NumOfProducts',
                  'IsActiveMember',
                  'EstimatedSalary',
                  'HasCrCard']

def preprocessing_pipeline():
    """
    Retorna uma pipeline de pré-processamento completa que pode ser usada diretamente no pipeline do imblearn.
    """
    return ColumnTransformer(
        transformers=[
            # Adicionando transformações personalizadas
            ('kmeans', KMeansClusterer(
                features=["CustomerId", "EstimatedSalary", "Balance"],
                n_clusters=10,
                random_state=123,
                n_components=3
            ), ['CustomerId', 'EstimatedSalary', 'Balance']),

            ('balance_salary_ratio', SalaryBalanceRatioTransformer(), ['EstimatedSalary', 'Balance']),
            ('balance_bin', VariableBinning(n_bins=9, column_name='Balance'), ['Balance']),
            ('credit_score_bin', VariableBinning(n_bins=5, column_name='CreditScore'), ['CreditScore']),
            ('estimated_salary_bin', VariableBinning(n_bins=25, column_name='EstimatedSalary'), ['EstimatedSalary']),
            ('age_bin', VariableBinning(n_bins=12, column_name='Age'), ['Age']),
            ('score_tier', CreditScoreTierTransformer(), ['CreditScore']),
            ('drop_columns', DropColumn(cols=['CustomerId', 'Surname', 'RowNumber']), ['CustomerId', 'Surname', 'RowNumber']),

            # Transformações categóricas
            ('encode_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_cols,),
            ('scaler', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough',
    )
