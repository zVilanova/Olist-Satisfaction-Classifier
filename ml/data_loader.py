import pandas as pd
from google.cloud import bigquery


def load_gold_features(project_id: str, dataset: str, table: str) -> pd.DataFrame:
    """Carrega a tabela de features da camada Gold do BigQuery para um DataFrame."""

    bq_client_gold = bigquery.Client(project=project_id)

    query = f"""
        SELECT * 
        FROM `{project_id}.{dataset}.{table}`
    """

    df = bq_client_gold.query(query).to_dataframe()

    return df
