WITH source_items AS (
    SELECT * FROM {{ ref('stg_olist_order_items') }}
),

cleaned_items AS (
    SELECT
        order_id,
        order_item_id,
        product_id,
        seller_id,
        shipping_limit_date,
        COALESCE(price, 0.0) AS price,
        COALESCE(freight_value, 0.0) AS freight_value
    FROM 
        source_items
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id, order_item_id
        ORDER BY shipping_limit_date DESC
    ) = 1
)

SELECT * FROM cleaned_items