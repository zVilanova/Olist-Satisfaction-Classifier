import matplotlib.pyplot as plt
import shap
from sklearn.metrics import classification_report, roc_auc_score


def print_metrics(model, X_test, y_test, title="Desempenho do Modelo"):
    """Gera o relatório de classificação e o score ROC-AUC."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {title} ---")
    print(classification_report(y_test, predictions))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")


def plot_shap_values(model, X_test):
    """Renderiza os gráficos de impacto global e individual das features."""
    explainer = shap.TreeExplainer(model)

    # Visão Global
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    plt.title("Impacto Global das Features na Insatisfação", fontsize=14)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.show()

    # Visão Individual (Waterfall)
    shap_obj = explainer(X_test)
    shap.plots.waterfall(shap_obj[0])
