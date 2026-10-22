WITH source_data AS (
    SELECT * FROM {{ source('bronze', 'olist_customers') }}
),

renamed_and_casted AS (
    SELECT
        TRIM(CAST(customer_id AS STRING)) AS customer_id,
        TRIM(CAST(customer_unique_id AS STRING)) AS customer_unique_id,
        LPAD(TRIM(CAST(customer_zip_code_prefix AS STRING)), 5, '0') AS customer_zip_code_prefix,
        TRIM(CAST(customer_city AS STRING)) AS customer_city,
        UPPER(TRIM(CAST(customer_state AS STRING))) AS customer_state
    FROM source_data
)

SELECT * FROM renamed_and_casted