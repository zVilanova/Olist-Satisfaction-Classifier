WITH source_customers AS (
    SELECT * FROM {{ ref('stg_olist_customer') }}
),

cleaned_customers AS (
    SELECT
        customer_id,
        customer_unique_id,
        customer_zip_code_prefix,
        customer_city,
        customer_state
    FROM 
        source_customers
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY customer_id
        ORDER BY customer_id
    ) = 1
)

SELECT * FROM cleaned_customers