import os
import tempfile

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = f"{PROJECT_ID}.olist_bronze"


def pipeline_bronze():
    print("Iniciando Pipeline de Ingestão (Bronze)")

    bq_client = bigquery.Client(project=PROJECT_ID)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    # O bloco abaixo cria uma pasta virtual temporaria
    with tempfile.TemporaryDirectory() as temp_dir:
        print("\nCriando o diretório temporário")

        # Faz o dowload do ZIP do kaggle direto para a pasta virtual temporaria e ja descompacta
        print("\nBaixando o Dataset")
        kaggle_api.dataset_download_files(
            "olistbr/brazilian-ecommerce", path=temp_dir, unzip=True
        )

        print("\nIniciando Streaming para o BigQuery")
        # Le tudo que foi descompactado na pasta temporaria
        for arquivo in os.listdir(temp_dir):
            if arquivo.endswith(".csv"):
                nome_tabela = arquivo.replace(".csv", "").replace("_dataset", "")
                tabela_destino = f"{DATASET_ID}.{nome_tabela}"
                caminho_csv = os.path.join(temp_dir, arquivo)
                caminho_parquet = os.path.join(temp_dir, f"{nome_tabela}.parquet")

                print(f"\nProcessando: {nome_tabela}")

                df = pd.read_csv(caminho_csv, dtype=str)
                df.to_parquet(caminho_parquet, engine="pyarrow", index=False)
                print("   -> Convertido para Parquet com sucesso")

                # Configura a carga no BigQuery indicando a fonte PARQUET
                job_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Substitui a tabela se ela ja existir
                )

                # Abre o CSV e envia para o BigQuery
                with open(caminho_parquet, "rb") as f_parquet:
                    job = bq_client.load_table_from_file(
                        f_parquet, tabela_destino, job_config=job_config
                    )

                job.result()  # Aguarda a confirmação de conclusão do BigQuery

    print("\nPipeline Finalizado")


if __name__ == "__main__":
    pipeline_bronze()
