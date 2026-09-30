from sklearn.model_selection import train_test_split
import pandas as pd

# PREPARAÇÃO DOS DADOS PARA EXPORTAÇÃO (CSV)
# parse_dates força o Pandas a converter o texto de volta para data
df_modelo = pd.read_csv('data/dados_logistica_silver.csv', parse_dates=['order_purchase_timestamp'])

df_modelo['mes_compra'] = df_modelo['order_purchase_timestamp'].dt.month

# Remoção de nulos
df_exportar = df_modelo[['customer_state', 'mes_compra', 'tempo_entrega_dias']].dropna()

# Divisão dos dados: 80% Treino e 20% Teste
df_treino, df_teste = train_test_split(df_exportar, test_size=0.20, random_state=42)

# Exportação
df_treino.to_csv('data/olist_treino.csv', index=False)
df_teste.to_csv('data/olist_teste.csv', index=False)

print(f"\nTreino: {len(df_treino)} linhas (80%)\nTeste: {len(df_teste)} linhas (20%)")