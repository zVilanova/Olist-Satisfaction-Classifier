WITH source_sellers AS (
    SELECT * FROM {{ ref('stg_olist_sellers') }}
),

cleaned_sellers AS (
    SELECT
        seller_id,
        seller_zip_code_prefix,
        seller_city,
        seller_state
    FROM source_sellers
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY seller_id 
        ORDER BY seller_id
    ) = 1
)

SELECT * FROM cleaned_sellers