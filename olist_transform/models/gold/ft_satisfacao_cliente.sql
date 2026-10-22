WITH orders AS (
    SELECT * FROM {{ ref('fct_orders') }}
),

order_items AS (
    SELECT * FROM {{ ref('fct_order_items') }}
),

reviews AS (
    SELECT * FROM {{ ref('fct_reviews') }}
),

products AS (
    SELECT * FROM {{ ref('dim_products') }}
),

customers AS (
    SELECT * FROM {{ ref('dim_customers') }}
),

sellers AS (
    SELECT * FROM {{ ref('dim_sellers') }}
),

payments AS (
    SELECT * FROM {{ ref('dim_payments') }}
),

payments_aggregated AS (
    SELECT
        order_id,
        MAX(payment_installments) AS qtd_parcelas
    FROM payments
    GROUP BY order_id
),

order_items_aggregated AS (
    SELECT
        order_id,
        MAX(order_item_id) AS qtd_itens_pedido,
        SUM(price) AS valor_total_produtos,
        SUM(freight_value) AS valor_total_frete
    FROM order_items
    GROUP BY order_id
),

joined_data AS (
    SELECT
        o.order_id,
        
        -- Features Logísticas
        DATE_DIFF(DATE(o.order_delivered_customer_date), DATE(o.order_purchase_timestamp), DAY) AS tempo_entrega_dias,
        DATE_DIFF(DATE(o.order_delivered_customer_date), DATE(o.order_estimated_delivery_date), DAY) AS dias_atraso,
        DATE_DIFF(DATE(o.order_delivered_carrier_date), DATE(o.order_approved_at), DAY) AS tempo_postagem_vendedor,
        
        -- Features Financeiras
        SAFE_DIVIDE(i.freight_value, i.price) AS proporcao_frete,
        (oia.valor_total_produtos + oia.valor_total_frete) AS valor_total_pedido,
        COALESCE(pa.qtd_parcelas, 1) AS qtd_parcelas,
        
        -- Features do Produto
        p.product_photos_qty AS qtd_fotos,
        p.product_description_length AS tamanho_descricao,
        (p.product_length_cm * p.product_height_cm * p.product_width_cm) AS volume_produto_cm3,
        oia.qtd_itens_pedido,
        
        -- Feature Geográfica
        IF(c.customer_state = s.seller_state, 1, 0) AS mesma_uf,
        
        -- Target Binário (1 = Satisfeito [>= 4], 0 = Insatisfeito [< 4])
        IF(r.review_score >= 4, 1, 0) AS target_satisfeito
        
    FROM orders o
    INNER JOIN order_items i ON o.order_id = i.order_id
    INNER JOIN reviews r ON o.order_id = r.order_id
    INNER JOIN products p ON i.product_id = p.product_id
    INNER JOIN customers c ON o.customer_id = c.customer_id
    INNER JOIN sellers s ON i.seller_id = s.seller_id
    LEFT JOIN payments_aggregated pa ON o.order_id = pa.order_id
    LEFT JOIN order_items_aggregated oia ON o.order_id = oia.order_id
)

SELECT * FROM joined_data
WHERE tempo_entrega_dias IS NOT NULL
  AND dias_atraso IS NOT NULL
  AND tempo_postagem_vendedor IS NOT NULL