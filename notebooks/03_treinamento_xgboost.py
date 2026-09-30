from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    recall_score,
    classification_report,
)
import xgboost as xgb
import pandas as pd
import shap
import matplotlib.pyplot as plt

# IMPORTAÇÃO DOS DADOS BRUTOS
colunas_data = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_delivered_carrier_date",
    "order_approved_at",
]

df_orders = pd.read_csv("data/olist_orders_dataset.csv", parse_dates=colunas_data)
df_items = pd.read_csv(
    "data/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]
)
df_reviews = pd.read_csv(
    "data/olist_order_reviews_dataset.csv",
    parse_dates=["review_answer_timestamp", "review_creation_date"],
)
df_products = pd.read_csv("data/olist_products_dataset.csv")
df_payments = pd.read_csv("data/olist_order_payments_dataset.csv")
df_customers = pd.read_csv("data/olist_customers_dataset.csv")
df_sellers = pd.read_csv("data/olist_sellers_dataset.csv")

# CRIAÇÃO DA TABELA GERAL
df_merged = pd.merge(df_orders, df_items, on="order_id", how="inner")
df_merged = pd.merge(df_merged, df_reviews, on="order_id", how="inner")
df_merged = pd.merge(df_merged, df_products, on="product_id", how="inner")
df_merged = pd.merge(df_merged, df_payments, on="order_id", how="inner")
df_merged = pd.merge(
    df_merged,
    df_customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="inner",
)
df_merged = pd.merge(
    df_merged, df_sellers[["seller_id", "seller_state"]], on="seller_id", how="inner"
)

df_merged = df_merged[df_merged["order_status"] == "delivered"].copy()

# CRIAÇÃO DAS FEATURES
# 1. Features Logísticas
df_merged["tempo_entrega_dias"] = (
    df_merged["order_delivered_customer_date"] - df_merged["order_purchase_timestamp"]
).dt.days
df_merged["dias_atraso"] = (
    df_merged["order_delivered_customer_date"]
    - df_merged["order_estimated_delivery_date"]
).dt.days
df_merged["tempo_postagem_vendedor"] = (
    df_merged["order_delivered_carrier_date"] - df_merged["order_approved_at"]
).dt.days

# 2. Features Financeiras
df_merged["proporcao_frete"] = df_merged["freight_value"] / df_merged["price"]
df_merged["valor_total_pedido"] = df_merged.groupby("order_id")["price"].transform(
    "sum"
) + df_merged.groupby("order_id")["freight_value"].transform("sum")
df_merged["qtd_parcelas"] = df_merged.groupby("order_id")[
    "payment_installments"
].transform("max")

# 3. Features do Produto e Anúncio
df_merged["qtd_fotos"] = df_merged["product_photos_qty"]
df_merged["tamanho_descricao"] = df_merged["product_description_lenght"]
df_merged["volume_produto_cm3"] = (
    df_merged["product_length_cm"]
    * df_merged["product_height_cm"]
    * df_merged["product_width_cm"]
)
df_merged["qtd_itens_pedido"] = df_merged.groupby("order_id")[
    "order_item_id"
].transform("max")

# 4. Features Geográficas
df_merged["mesma_uf"] = (
    df_merged["customer_state"] == df_merged["seller_state"]
).astype(int)

# Definindo as features (X) e o target (y)
features = [
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

X = df_merged[features].dropna()
y = df_merged.loc[X.index, "review_score"]

# Transformando o target em binário (1 = Satisfeito, 0 = Insatisfeito)
y_binario = y.apply(lambda x: 1 if x >= 4 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binario, test_size=0.2, random_state=42, stratify=y_binario
)

# Cálculo do peso para lidar com o desbalanceamento (dar foco aos insatisfeitos)
peso_classes = (y_train == 0).sum() / (y_train == 1).sum()

# Instanciação e Treinamento do Modelo
xgb_model = xgb.XGBClassifier(
    random_state=42, eval_metric="logloss", scale_pos_weight=peso_classes
)

xgb_model.fit(X_train, y_train)

# Previsões e Avaliação de Desempenho
previsoes = xgb_model.predict(X_test)
probabilidades = xgb_model.predict_proba(X_test)[:, 1]

print("--- Relatório de Classificação ---")
print(classification_report(y_test, previsoes))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilidades):.4f}")

# CRIAÇÃO DOS GRÁFICOS DE COMPORTAMENTO DO MODELO (SHAP)
# Inicializar o interpretador do SHAP com o modelo treinado
explainer = shap.TreeExplainer(xgb_model)

# O SHAP testa exaustivamente como cada variável afeta a previsão de cada linha
shap_values = explainer.shap_values(X_test)

# Visão Global (Summary Plot) -> Mostra quais variáveis mais influenciam o modelo no panorama geral
plt.title("Impacto Global das Features na Insatisfação", fontsize=14)
shap.summary_plot(shap_values, X_test)

# Visão Individual (Waterfall Plot) -> Mostra exatamente o que levou o modelo a prever a nota para o 1º cliente do teste
shap_obj = explainer(X_test)
shap.plots.waterfall(shap_obj[0])

# OTIMIZAÇÃO DO RECALL DO MODELO COM HIPERPARÂMETROS
# Alvo da otimização: Maximizar o Recall da classe 0 (Insatisfeitos)
scorer_foco = make_scorer(recall_score, pos_label=0)

# Opções que o Python vai cruzar e testar
param_grid = {
    "n_estimators": [100, 200, 300],  # Número de árvores
    "max_depth": [3, 4, 5, 6, 7],  # Profundidade máxima do raciocínio
    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Velocidade com que corrige os erros
    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],  # porcentagem de dados aleatórios vistos por árvore
    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],  # porcentagem de features aleatórias vistas por árvore
}

# Instanciar o modelo base
xgb_base = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=peso_classes,  # Variável balanceadora
)

# Cross-Validation
cv_estratificado = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

busca_aleatoria = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=15,  # Limita a 15 testes para não congelar
    scoring=scorer_foco,  # Avaliador da classe 0
    cv=cv_estratificado,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

# Treino do modelo ajustado
print("Iniciando a busca pela melhor calibração...")
busca_aleatoria.fit(X_train, y_train)

# 6. Extrair e avaliar o modelo vencedor
melhor_xgb = busca_aleatoria.best_estimator_

print("\n--- Melhor Configuração Encontrada ---")
print(busca_aleatoria.best_params_)

print("\n--- Desempenho do Novo Modelo ---")
previsoes_otimizadas = melhor_xgb.predict(X_test)
print(classification_report(y_test, previsoes_otimizadas))

# Guardar o modelo no formato nativo
melhor_xgb.save_model("data/modelo_olist_xgb.json")
print("Modelo guardado com sucesso em JSON!")
