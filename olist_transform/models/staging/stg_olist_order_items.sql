WITH source_data AS (
    SELECT * FROM {{ source('bronze', 'olist_order_items') }}
),

renamed_and_casted AS (
    SELECT
        TRIM(CAST(order_id AS STRING)) AS order_id,
        SAFE_CAST(order_item_id AS INT64) AS order_item_id,
        TRIM(CAST(product_id AS STRING)) AS product_id,
        TRIM(CAST(seller_id AS STRING)) AS seller_id,
        SAFE_CAST(shipping_limit_date AS TIMESTAMP) AS shipping_limit_date,
        SAFE_CAST(price AS NUMERIC) AS price,
        SAFE_CAST(freight_value AS NUMERIC) AS freight_value
    FROM source_data
)

SELECT * FROM renamed_and_casted