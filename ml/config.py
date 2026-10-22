import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_GOLD = "olist_gold"
TABLE_GOLD = "ft_satisfacao_cliente"

# Features e Target
FEATURES = [
    "tempo_entrega_dias",
    "dias_atraso",
    "tempo_postagem_vendedor",
    "proporcao_frete",
    "valor_total_pedido",
    "qtd_parcelas",
    "qtd_fotos",
    "tamanho_descricao",
    "volume_produto_cm3",
    "qtd_itens_pedido",
    "mesma_uf",
]
TARGET = "target_satisfeito"

# Parametros de Treino e Divisao
TEST_SIZE = 0.2
RANDOM_STATE = 42
EVAL_METRIC = "logloss"
N_SPLITS_CV = 3
N_ITER = 15

# Opcoes de cruzamento para RandomizedSearchCV
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

