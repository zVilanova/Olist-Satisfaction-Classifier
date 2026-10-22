WITH source_data AS (
    SELECT * FROM {{ source('bronze', 'olist_order_reviews') }}
),

renamed_and_casted AS (
    SELECT
        TRIM(CAST(review_id AS STRING)) AS review_id,
        TRIM(CAST(order_id AS STRING)) AS order_id,
        SAFE_CAST(review_score AS INT64) AS review_score,
        TRIM(CAST(review_comment_title AS STRING)) AS review_comment_title,
        TRIM(CAST(review_comment_message AS STRING)) AS review_comment_message,
        SAFE_CAST(review_creation_date AS TIMESTAMP) AS review_creation_date,
        SAFE_CAST(review_answer_timestamp AS TIMESTAMP) AS review_answer_timestamp
    FROM source_data
)

SELECT * FROM renamed_and_casted