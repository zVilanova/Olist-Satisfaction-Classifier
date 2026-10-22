WITH source_payments AS (
    SELECT * FROM {{ ref('stg_olist_order_payments') }}
),

cleaned_payments AS (
    SELECT
        order_id,
        payment_sequential,
        COALESCE(payment_type, 'nao_definido') AS payment_type,
        COALESCE(payment_installments, 1) AS payment_installments,
        COALESCE(payment_value, 0.0) AS payment_value
    FROM source_payments
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id, payment_sequential 
        ORDER BY payment_sequential ASC
    ) = 1
)

SELECT * FROM cleaned_payments