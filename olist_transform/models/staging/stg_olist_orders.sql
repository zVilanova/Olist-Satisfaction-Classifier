WITH source_data AS (
    SELECT * FROM {{ source('bronze', 'olist_orders') }}
),

renamed_and_casted AS (
    SELECT
        -- PK, FK & Text Attributes
        TRIM(CAST(order_id AS STRING)) AS order_id,
        TRIM(CAST(customer_id AS STRING)) AS customer_id,
        LOWER(TRIM(CAST(order_status AS STRING))) AS order_status,
        -- Timestamps
        SAFE_CAST(order_purchase_timestamp AS TIMESTAMP) AS order_purchase_timestamp,
        SAFE_CAST(order_approved_at AS TIMESTAMP) AS order_approved_at,
        SAFE_CAST(order_delivered_carrier_date AS TIMESTAMP) AS order_delivered_carrier_date,
        SAFE_CAST(order_delivered_customer_date AS TIMESTAMP) AS order_delivered_customer_date,
        SAFE_CAST(order_estimated_delivery_date AS TIMESTAMP) AS order_estimated_delivery_date
    FROM source_data
)

SELECT * FROM renamed_and_casted