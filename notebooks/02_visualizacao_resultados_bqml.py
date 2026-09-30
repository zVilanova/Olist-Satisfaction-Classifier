import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Resultados gerados pelo BigQuery
df_grafico = pd.read_csv('data/resultado_modelo_bq.csv')

# Agrupando para calcular a média Real vs Prevista por estado
df_medias = df_grafico.groupby('customer_state')[['valor_real', 'valor_previsto']].mean().round(1).reset_index()

# Ordenando pelo tempo real
df_medias = df_medias.sort_values('valor_real')

fig, ax = plt.subplots(figsize=(12, 10))
y = np.arange(len(df_medias['customer_state'])) # Posições no eixo Y para os 27 estados
altura_barra = 0.35

# Barras: Azul para o Real, Vermelho para o Previsto
ax.barh(y - altura_barra/2, df_medias['valor_real'], height=altura_barra, label='Tempo Real (Dias)', color='#4C72B0')
ax.barh(y + altura_barra/2, df_medias['valor_previsto'], height=altura_barra, label='Previsão BQML (Dias)', color='#C44E52')

ax.set_yticks(y)
ax.set_yticklabels(df_medias['customer_state'])
ax.invert_yaxis() # Mantém os estados mais rápidos no topo

plt.title('Tempo Médio Real vs Previsão BQML por Estado', fontsize=14, pad=15)
plt.xlabel('Quantidade de Dias', fontsize=12)
plt.ylabel('Estado (UF)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()