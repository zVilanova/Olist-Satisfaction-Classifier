import config
import evaluate
import xgboost as xgb
from data_loader import load_gold_features
from sklearn.model_selection import train_test_split


def run_train():
    # Carregamento dos dados injetando os parametros de config
    df = load_gold_features(
        project_id=config.PROJECT_ID,
        dataset=config.DATASET_GOLD,
        table=config.TABLE_GOLD,
    )

    # Definicao das Features (X) e Target (y)
    X = df[config.FEATURES].astype(float)
    y_binario = df[config.TARGET]

    # Separacao de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binario,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_binario,
    )

    # Calculo do peso (classes) para lidar com desbalanceamento
    peso_classes = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_base = xgb.XGBClassifier(
        random_state=config.RANDOM_STATE,
        eval_metric=config.EVAL_METRIC,
        scale_pos_weight=peso_classes,
    )
    xgb_base.fit(X_train, y_train)

    xgb_base.save_model("artifacts/modelo_olist_xgb_baseline.json")

    evaluate.print_metrics(xgb_base, X_test, y_test, title="Desempenho do Modelo BASE")


if __name__ == "__main__":
    run_train()
