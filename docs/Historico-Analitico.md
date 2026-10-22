# 📜 Histórico Analítico · A história por trás do projeto

> **Este documento preserva a Fase 1 do projeto:** a pesquisa em análise de dados e modelagem que deu origem ao modelo em produção.  
> A arquitetura atual (Airflow, Docker e Streamlit) está descrita no [README principal](../README.md).  
> Os scripts, queries e gráficos originais desta fase não fazem parte do repositório atual, que contém apenas a arquitetura de produção. Este documento preserva a **metodologia, as decisões e os resultados**.

Esta fase conta a história em três atos: a análise da logística de entrega, a estimativa de prazos com BigQuery ML e a previsão da satisfação do cliente com um modelo XGBoost.

| Ato | Pergunta | Ferramenta | Resultado |
|---|---|---|---|
| **1** | Quanto tempo a entrega leva em cada estado? | Python · Pandas | SP: **8,3 dias** × RR: **29,0 dias** |
| **2** | Dá para prever o prazo só com estado (UF) e mês? | BigQuery ML · Regressão Linear | **R² ~0,24**, **MAE 5,3 dias** |
| **3** | Dá para identificar pedidos com risco de insatisfação? | XGBoost · SHAP | **Recall 57%** · **Precisão 43%** · **Acurácia 72%** |

## 🧭 Sumário

- [🎲 Contexto e dados](#-contexto-e-dados)
- [Ato 1 · Prazo de entrega por estado](#ato-1--quanto-tempo-o-pedido-leva-para-chegar-em-cada-estado)
- [Ato 2 · Previsão de prazo com BigQuery ML](#ato-2-bqml--é-possível-prever-quantos-dias-um-pedido-demorará-a-chegar)
- [Ato 3 · Previsão de satisfação com XGBoost](#ato-3-xgboost--como-antecipar-a-insatisfação-do-cliente-antes-da-avaliação)
- [➡️ E depois? A virada para MLOps](#️-e-depois-a-virada-para-mlops)

---

## 🎲 Contexto e Dados

Base: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Este dataset reúne dados anonimizados de aproximadamente 100 mil pedidos realizados no Brasil entre 2016 e 2018 por meio da Olist, uma plataforma de e-commerce que conecta pequenos lojistas a diferentes marketplaces. Os dados são provenientes de informações comerciais reais, porém foram anonimizados para preservar a privacidade dos envolvidos: informações que poderiam identificar empresas e parceiros presentes nos textos das avaliações foram substituídas por nomes de casas de **Game of Thrones**.

<details>
<summary>Tabelas utilizadas</summary>

Foram utilizadas 7 tabelas:
- `olist_orders_dataset`
- `olist_customers_dataset`
- `olist_order_payments_dataset`
- `olist_order_reviews_dataset`
- `olist_order_items_dataset`
- `olist_products_dataset`
- `olist_sellers_dataset`

Elas representam dados sobre pedidos, clientes, pagamentos, avaliações, itens, produtos e vendedores.

</details>

---

## Ato 1 · Quanto tempo o pedido leva para chegar em cada estado?

**Resultado principal:**
- O estado de São Paulo (SP) possui a entrega mais rápida, com média de **8,3 dias**.
- No extremo oposto, Roraima (RR) apresenta o maior tempo de espera, com média de **29,0 dias**.

**Interpretação:** a diferença de prazos é grande: a entrega em Roraima leva mais de três vezes o tempo de São Paulo, e o Centro-Oeste apresenta tempos intermediários. Uma hipótese é que os centros de distribuição e a maioria dos vendedores estejam concentrados no Sul e Sudeste, o que acelera e barateia o frete nessas regiões, enquanto os estados do Norte (como RR, AP e AM) e do Nordeste ficam mais distantes desses centros e podem enfrentar desafios de infraestrutura, como o transporte pela bacia amazônica. Os dados mostram o padrão, mas não testam essas causas.

<details>
<summary>Como o tempo de entrega foi calculado</summary>

### Cálculo realizado

- **Arquivos e variáveis:** `olist_orders_dataset` (`order_status`, `order_purchase_timestamp`, `order_delivered_customer_date`) e `olist_customers_dataset` (`customer_id`, `customer_state`).
- **Subconjunto analisado:** apenas pedidos com status `delivered` (entregues).
- **Cálculo:** o tempo de entrega foi obtido pela subtração entre a data de compra e a data de entrega efetiva (`dt.days`). Em seguida, os dados foram agrupados por estado (`customer_state`), calculando-se a média de dias, ordenada de forma crescente.

</details>

---

## Ato 2 (BQML) · É possível prever quantos dias um pedido demorará a chegar?

- **Objetivo de negócio:** prever a quantidade de dias necessários para a entrega de um pedido, utilizando o estado de destino e a sazonalidade (mês da compra) como variáveis preditoras em um modelo de Machine Learning no BigQuery (BQML).
- **Metodologia:** os dados históricos foram exportados para o Google Cloud Platform (BigQuery ML). Optou-se por uma **Regressão Linear**, adequada para a previsão de variáveis contínuas. A variável `mes_compra` foi convertida em texto (STRING) para que o algoritmo a interpretasse como fator categórico de sazonalidade via One-Hot Encoding.

<details>
<summary>Como funciona a regressão linear do BQML?</summary>

O BQML utiliza a Regressão Linear para prever um valor numérico com base em outras informações. Você fornece ao modelo os dados históricos, indicando quais variáveis podem ajudar na previsão (*features*) e qual valor deseja prever (*label*). Ele encontra uma relação matemática entre as variáveis de entrada e o resultado. Depois de treinado, você fornece novos dados e ele retorna uma previsão numérica.

</details>

### Metodologia em 4 etapas

**Etapa 1 · Dataset de exportação**  
A partir do dataset original no Pandas, as colunas de estado de destino e tempo de entrega foram isoladas e uma nova variável preditora de sazonalidade (mês da compra) foi gerada. Para garantir o rigor estatístico e evitar overfitting, a base limpa foi dividida aleatoriamente com scikit-learn:

- 80% dos registros foram destinados ao treinamento do modelo
- 20% foram isolados exclusivamente para teste

Esses subconjuntos foram exportados como arquivos CSV distintos e carregados no BigQuery.

**Etapa 2 · Treinamento · `CREATE MODEL`**  
O comando `CREATE MODEL` instrui o BigQuery a processar os dados históricos e salvar os padrões encontrados. O bloco `OPTIONS` é a configuração principal:

- `model_type` define o algoritmo (Regressão Linear, ideal para prever grandezas contínuas como dias)
- `input_label_cols` aponta a coluna "gabarito" que o modelo deve aprender

O modelo foi treinado ignorando registros nulos em que a data de entrega não foi preenchida.

**Etapa 3 · Avaliação · `ML.EVALUATE`**  
A função testa a precisão do modelo comparando suas previsões com a realidade histórica, sempre sobre a base de teste. As métricas analisadas foram:

- **MAE (Erro Médio Absoluto):** quantifica a margem de erro na unidade original (dias)
- **R² (R-quadrado):** indica qual porcentagem da variação do tempo de entrega é explicada pelas variáveis fornecidas

**Resultado:** MAE de **5,33 dias** e R² de **0,2398**. Isso indica que a geografia e a sazonalidade explicam cerca de 24% das variações no tempo de entrega, e que a previsão erra, em média, mais de 5 dias. É um resultado modesto: serve como **ponto de partida (baseline)** e mostra que localização e sazonalidade, sozinhas, não bastam. Variáveis como origem do vendedor e dimensões do pacote são candidatas a reduzir esse erro, e o Ato 3 passa a utilizá-las, agora para prever a satisfação.

```json
[{
  "mean_absolute_error": "5.3340601450903442",
  "mean_squared_error": "65.956713010218863",
  "mean_squared_log_error": "0.31213105797074003",
  "median_absolute_error": "3.9446816497247852",
  "r2_score": "0.23986783484599683",
  "explained_variance": "0.24001579696832842"
}]
```

**Etapa 4 · Inferência · `ML.PREDICT`**  
A função aplica os padrões aprendidos a novas transações, gerando a coluna `predicted_tempo_entrega_dias`. Foi aplicada sobre a base de teste para prever o prazo linha a linha, e o resultado foi exportado de volta ao Python para plotar o comparativo entre valor real e estimativa do BigQuery.

**Interpretação:** a comparação por estado (UF) mostra onde o modelo funciona melhor e pior. Nas regiões Sul, Sudeste e Centro-Oeste, as previsões ficaram mais próximas do prazo real. Na região Norte (RR, AM, AC), o modelo subestimou os prazos de forma consistente. Uma hipótese é que nessas áreas pesem fatores que o modelo não enxerga, como a distância até o vendedor e o tipo de transporte (fluvial, por exemplo), mas isso não foi testado aqui. Com apenas estado e mês, o modelo não tem como capturar essas diferenças, o que exige features adicionais.

---

## Ato 3 (XGBoost) · Como antecipar a insatisfação do cliente antes da avaliação?

- **Objetivo de negócio:** avaliar simultaneamente múltiplos parâmetros da jornada de compra (tempo total de entrega, dias de atraso, características do anúncio etc.) para identificar e ranquear quais variáveis mais influenciam a nota de avaliação (1 a 5 estrelas) dada pelo consumidor, direcionando os esforços de melhoria da empresa.
- **Metodologia:** classificação binária com **XGBoost (Extreme Gradient Boosting)** em Python, escolhido por ser um dos algoritmos mais potentes para dados tabulares e por lidar bem com relações não lineares. A variável alvo foi criada a partir da nota da avaliação (`review_score`), classificando os pedidos em "Satisfeito" (notas altas) e "Insatisfeito" (notas baixas).

<details>
<summary>Como funciona o XGBoost?</summary>

Diferente da regressão linear do Ato 2, o XGBoost cria um "conjunto de árvores de decisão".

- **O que é uma árvore de decisão?** Um fluxograma de perguntas e respostas. A primeira ramificação pode ser *"O pedido atrasou?"*. Se sim, segue por um caminho; se não, por outro. Em seguida pode perguntar *"O frete custou mais de 20% do valor do produto?"*, até chegar à classificação final: Satisfeito ou Insatisfeito.
- **A lógica do Gradient Boosting:** o modelo não cria apenas uma árvore, mas centenas, de forma sequencial. Cada nova árvore é treinada com o objetivo específico de **corrigir os erros** das anteriores, como uma equipe em que cada novo membro conserta os pontos cegos dos anteriores.

</details>

### Metodologia em 4 etapas

**Etapa 1 · Engenharia de features e preparação da base**  
O Ato 2 evidenciou que a localização e a sazonalidade, sozinhas, não representam todo o cenário. Para suprir essa lacuna, foi feito o cruzamento de dados (JOIN) para consolidar a jornada de compra em uma única base. As features criadas:

| Categoria | Variável | Descrição |
| :--- | :--- | :--- |
| **Logística** | `tempo_entrega_dias` | Total de dias corridos entre a compra e a entrega ao cliente. |
| | `dias_atraso` | Diferença em dias entre a data de entrega estimada e a data real. |
| | `tempo_postagem_vendedor` | Tempo que o vendedor levou para despachar o produto após a aprovação da compra. |
| **Financeira** | `proporcao_frete` | Valor do frete em relação ao valor do item. |
| | `valor_total_pedido` | Soma do valor dos produtos e do frete. |
| | `qtd_parcelas` | Número de parcelas escolhidas no momento do pagamento. |
| **Produto/Anúncio** | `qtd_fotos` | Quantidade de imagens disponibilizadas no anúncio do produto. |
| | `tamanho_descricao` | Número de caracteres da descrição do produto. |
| | `volume_produto_cm3` | Volume estimado da embalagem (comprimento × largura × altura). |
| | `qtd_itens_pedido` | Quantidade total de itens físicos dentro do mesmo pedido. |
| **Geográfica** | `mesma_uf` | Indica (1 ou 0) se cliente e vendedor estão no mesmo estado. |

A base enriquecida foi dividida aleatoriamente com scikit-learn: **80% para treino** e **20% para teste**.

**Etapa 2 · Matriz de preditoras e variável alvo**  
- **Matriz de preditoras (X):** as 11 variáveis de engenharia de features.
- **Variável alvo (y):** `review_score` transformado em duas classes. Notas 1, 2 e 3 viram `0` (Insatisfeito); notas 4 e 5 viram `1` (Satisfeito).

> 💡 *Por que essa decisão faz sentido para o negócio?* No contexto de Customer Experience (CX), a diferença prática entre uma nota 1 e 2 é mínima: ambas indicam um atrito severo e um cliente detrator. O objetivo da área de atendimento não é prever o número exato de estrelas, mas identificar se a experiência de compra foi frustrante o suficiente para exigir uma ação de mitigação.

O `XGBoost Classifier (Binary)` foi treinado e avaliado contra a base de teste (dados inéditos para o algoritmo):

```text
              precision    recall  f1-score   support

           0       0.47      0.57      0.51      5248
           1       0.86      0.80      0.83     17394

    accuracy                           0.75     22642
   macro avg       0.66      0.69      0.67     22642
weighted avg       0.77      0.75      0.76     22642

ROC-AUC Score: 0.7481
```

- **Acurácia (75%):** o modelo acerta o sentimento final do cliente em 3 a cada 4 pedidos.
- **Identificação de clientes em risco (Classe 0):** o *Recall* de 57% para a classe minoritária significa que a empresa consegue interceptar proativamente quase 60% dos clientes que teriam uma experiência ruim, antes mesmo de registrarem a reclamação.
- **ROC-AUC (~0,75):** indica uma capacidade sólida de distinguir os padrões que diferenciam um cliente promotor de um detrator.

**Etapa 3 · Explicabilidade com SHAP**  
Modelos baseados em árvores costumam ser vistos como "caixas pretas". Para que o time de negócios confiasse nas previsões, foi usada a biblioteca **SHAP (Shapley Additive Explanations)**, baseada na Teoria dos Jogos.

- **Summary Plot (visão global):** mostra quais variáveis mais impactam o modelo de forma geral.

  ![SHAP Summary](../artifacts/shap_summary.png)

  *(Versão atual do gráfico, gerada automaticamente pelo pipeline de treino.)*

  > **💡 Insight:** as variáveis logísticas dominam o topo. Um `dias_atraso` alto (pontos vermelhos à direita) empurra a previsão fortemente para a insatisfação (Classe 0).

- **Waterfall Plot (visão individual):** destrincha **um único pedido**. O padrão de cores pode parecer contraintuitivo: as barras **vermelhas** empurram o valor numérico para cima, enquanto as **azuis** puxam para baixo. Esse gráfico hoje é gerado em tempo real pelo [app Streamlit](../README.md#️-aplicação-streamlit) para qualquer cenário simulado.

  > **💡 Como a área de CX usa isso?** O gráfico explica o "porquê" da previsão. Neste exemplo real, o modelo indica que o cliente ficará **Satisfeito**. A logística pesou a favor: o pedido foi entregue em apenas 5 dias (contribuição de +0,35) e chegou 5 dias antes do prazo prometido (`dias_atraso` = -5, +0,14), apesar do frete alto (62,2% do preço do produto).  
  > *Os valores do gráfico estão na escala interna do modelo (log-odds), não em pontos percentuais: um resultado final de 0,71 corresponde a cerca de 67% de probabilidade de satisfação.*

**Etapa 4 · Otimização de hiperparâmetros**  
Foi feito um ajuste fino do modelo com `RandomizedSearchCV`, variando `max_depth`, `learning_rate` e `n_estimators`, para evitar overfitting e garantir boa generalização.

- `max_depth`: controla a profundidade máxima das árvores, evitando um modelo excessivamente complexo
- `learning_rate` e `n_estimators`: equilibram a velocidade de aprendizado e a quantidade de árvores construídas

**Resultado da otimização:** o *Recall* da classe insatisfeito ficou igual (57%), mas as demais métricas caíram um pouco em relação ao modelo inicial:

| Modelo | Precisão (insat.) | Recall (insat.) | F1 (insat.) | Acurácia | ROC-AUC |
|---|---|---|---|---|---|
| Inicial (parâmetros padrão) | 0,47 | 0,57 | 0,51 | 0,75 | 0,748 |
| Final (ajustado) | 0,43 | 0,57 | 0,49 | 0,72 | - |

Como o recall, métrica priorizada na busca, ficou igual nos dois modelos e as demais métricas variaram pouco, o modelo ajustado foi o escolhido para o app, uma vez que a busca não trouxe um ganho claro.

```text
Iniciando a busca pela melhor calibração...
Fitting 3 folds for each of 15 candidates, totalling 45 fits

--- Melhor Configuração Encontrada ---
{'subsample': 1.0, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.7}

--- Desempenho do Modelo ---
              precision    recall  f1-score   support

           0       0.43      0.57      0.49      5248
           1       0.86      0.77      0.81     17394

    accuracy                           0.72     22642
   macro avg       0.64      0.67      0.65     22642
weighted avg       0.76      0.72      0.74     22642
```

---

## ➡️ E depois? A virada para MLOps

Ao final desta fase, o modelo existia, mas vivia em scripts e notebooks, treinado "à mão". O passo seguinte foi transformá-lo em um sistema: pipeline de treino orquestrado, ambiente conteinerizado e aplicação desacoplada servindo as previsões.

👉 **Veja a arquitetura atual no [README principal](../README.md).**
