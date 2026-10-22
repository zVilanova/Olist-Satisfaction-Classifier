WITH source_data AS (
    SELECT * FROM {{ source('bronze', 'olist_order_payments') }}
),

renamed_and_casted AS (
    SELECT
        TRIM(CAST(order_id AS STRING)) AS order_id,
        SAFE_CAST(payment_sequential AS INT64) AS payment_sequential,
        TRIM(CAST(payment_type AS STRING)) AS payment_type,
        SAFE_CAST(payment_installments AS INT64) AS payment_installments,
        SAFE_CAST(payment_value AS NUMERIC) AS payment_value
    FROM source_data
)

SELECT * FROM renamed_and_casted