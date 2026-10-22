WITH source_products AS (
    SELECT * FROM {{ ref('stg_olist_products') }}
),

cleaned_products AS (
    SELECT 
        product_id,
        COALESCE(product_category_name, 'outros') AS product_category_name,
        COALESCE(product_name_length, 0) AS product_name_length,
        COALESCE(product_description_length, 0) AS product_description_length,
        COALESCE(product_photos_qty, 0) AS product_photos_qty,
        COALESCE(product_weight_g, 0.0) AS product_weight_g,
        COALESCE(product_length_cm, 0.0) AS product_length_cm,
        COALESCE(product_height_cm, 0.0) AS product_height_cm,
        COALESCE(product_width_cm, 0.0) AS product_width_cm
    FROM
        source_products
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY product_id
        ORDER BY product_id
    ) = 1
)

SELECT * FROM cleaned_products