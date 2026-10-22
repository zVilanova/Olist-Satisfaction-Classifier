WITH source_orders AS (
    SELECT * FROM {{ ref('stg_olist_orders') }}
),

filtered_orders AS (
    SELECT
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_estimated_delivery_date
    FROM 
        source_orders
    WHERE 
        order_status = 'delivered'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id
        ORDER BY order_purchase_timestamp DESC
    ) = 1
)

SELECT * FROM filtered_orders