WITH source_reviews AS (
    SELECT * FROM {{ ref('stg_olist_order_reviews') }}
),

deduplicated_reviews AS (
    SELECT
        review_id,
        order_id,
        review_score,
        review_comment_title,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp
    FROM 
        source_reviews
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id
        ORDER BY review_answer_timestamp DESC, review_creation_date DESC
    ) = 1
)

SELECT * FROM deduplicated_reviews