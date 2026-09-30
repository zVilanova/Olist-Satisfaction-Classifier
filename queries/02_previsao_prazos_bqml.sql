# CRIAÇÃO DO MODELO
CREATE OR REPLACE MODEL `seu_modelo`
OPTIONS(
  model_type='linear_reg',
  input_label_cols=['tempo_entrega_dias']
) AS
SELECT
  customer_state,
  CAST(mes_compra AS STRING) AS mes_compra,
  tempo_entrega_dias
FROM
  `sua_tabela_de_treino`;


# AVALIAÇÃO DO MODELO CRIADO
SELECT *
FROM ML.EVALUATE(
  MODEL `seu_modelo`,
  (
    SELECT
      customer_state,
      CAST(mes_compra AS STRING) AS mes_compra,
      tempo_entrega_dias
    FROM `sua_tabela_de_teste`
  )
);


# PREDIÇÃO UTILIZANDO O MODELO
SELECT
  customer_state,
  tempo_entrega_dias AS valor_real,
  ROUND(predicted_tempo_entrega_dias, 1) AS valor_previsto
FROM ML.PREDICT(
  MODEL `seu_modelo`,
  (
    SELECT
      customer_state,
      CAST(mes_compra AS STRING) AS mes_compra,
      tempo_entrega_dias
    FROM
      `sua_tabela_de_teste`
  )
);