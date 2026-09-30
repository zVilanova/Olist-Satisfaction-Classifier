import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import urllib.request
import json

# TRATAMENTO DOS DADOS
df_orders = pd.read_csv('data/olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date'])
df_customers = pd.read_csv('data/olist_customers_dataset.csv')

# pedidos que foram efetivamente entregues
df_entregues = df_orders[df_orders['order_status'] == 'delivered'].copy()

# tempo de entrega em dias (Data de Entrega - Data de Compra)
df_entregues['tempo_entrega_dias'] = (df_entregues['order_delivered_customer_date'] - df_entregues['order_purchase_timestamp']).dt.days

# join com a tabela de clientes para trazer a coluna de Estado
df_logistica = pd.merge(df_entregues, df_customers[['customer_id', 'customer_state']], on='customer_id', how='inner')

# df_logistica é utilizado em outras etapas, portanto salvamos os dados 
df_logistica.to_csv('data/dados_logistica_silver.csv', index=False)

# reset_index() transforma o resultado de volta em um dataframe normal
media_por_estado = df_logistica.groupby('customer_state')['tempo_entrega_dias'].mean().round(1).reset_index() 

# ordenando o resultado pelo `tempo_entrega_dias` crescente
media_por_estado = media_por_estado.sort_values(by='tempo_entrega_dias')


# CRIAÇÃO DO GRÁFICO
plt.figure(figsize=(10, 8))

# criando o gráfico de barras horizontais
plt.barh(media_por_estado['customer_state'], media_por_estado['tempo_entrega_dias'], color='#4C72B0')

# invertendo o eixo Y para que o mais rápido (SP) fique no topo do gráfico
plt.gca().invert_yaxis()

plt.title('Tempo Médio de Entrega por Estado (em dias)', fontsize=14, pad=15)
plt.xlabel('Dias até a entrega', fontsize=12)
plt.ylabel('Estado (UF)', fontsize=12)

plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()


# MALHA GEOGRÁFICA DO BRASIL (GeoJSON)
url_geojson = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

with urllib.request.urlopen(url_geojson) as response:
    brazil_geojson = json.load(response)

fig = px.choropleth(
    media_por_estado,                    
    geojson=brazil_geojson,              
    locations='customer_state',          
    featureidkey="properties.sigla",     
    color='tempo_entrega_dias',          
    color_continuous_scale="YlOrRd",    
    title="Tempo Médio de Entrega no Mapa do Brasil"
)

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

fig.show()