from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

CAMINHO_PROJETO = "C:/Users/PICHAU/Desktop/Olist-Refactor"

# Configurações padrão do agendamento
default_args = {
    'owner': 'engenharia_dados',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Definindo a DAG
with DAG(
    dag_id='olist_ml_pipeline',
    default_args=default_args,
    description='Orquestração semanal do modelo XGBoost - Olist',
    schedule_interval='0 2 * * 1', # Expressão Cron: Roda toda segunda-feira às 02:00 da manhã
    start_date=datetime(2023, 10, 1),
    catchup=False,
    tags=['mlops', 'xgboost', 'bigquery'],
) as dag:

    treinar_modelo = DockerOperator(
        task_id='treinar_modelo_baseline',
        image='olist-refactor-pipeline:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock', # Permite que o Airflow comande o Docker do Windows
        network_mode='bridge',
        
        environment={
            'GOOGLE_APPLICATION_CREDENTIALS': '/app/gcp-key.json',
            'GCP_PROJECT_ID': 'olist-project-511521' 
        },
        
        mounts=[
            Mount(source=f"{CAMINHO_PROJETO}/artifacts", target="/app/artifacts", type="bind"),
            Mount(source=f"{CAMINHO_PROJETO}/gcp-key.json", target="/app/gcp-key.json", type="bind")
        ],
    )
    
    treinar_modelo