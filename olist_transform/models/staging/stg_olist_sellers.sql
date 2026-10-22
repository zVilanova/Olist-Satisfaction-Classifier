WITH source_data AS (
    SELECT * FROM {{ source('bronze', 'olist_sellers') }}
),

renamed_and_casted AS (
    SELECT
        TRIM(CAST(seller_id AS STRING)) AS seller_id,
        LPAD(TRIM(CAST(seller_zip_code_prefix AS STRING)), 5, '0') AS seller_zip_code_prefix,
        TRIM(CAST(seller_city AS STRING)) AS seller_city,
        UPPER(TRIM(CAST(seller_state AS STRING))) AS seller_state
    FROM source_data
)

SELECT * FROM renamed_and_casted