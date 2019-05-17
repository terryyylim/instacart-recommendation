WITH product_info AS (
    SELECT
        products.product_id pid,
        products.product_name,
        aisles.aisle,
        departments.department
    FROM
        products 
    LEFT JOIN aisles ON products.aisle_id = aisles.aisle_id
    LEFT JOIN departments ON products.department_id = departments.department_id
)
, product_count AS (
    SELECT
        DISTINCT(product_id),
        COUNT(order_id) num
    FROM
       order_products_prior
    GROUP BY 1
)
, final AS (
    SELECT
        product_info.pid,
        product_info.product_name,
        product_info.aisle,
        product_info.department,
        product_count.num
    FROM
        product_info
    LEFT JOIN product_count ON product_info.pid = product_count.product_id
)

SELECT * FROM final