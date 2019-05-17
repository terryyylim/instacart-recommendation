SELECT
    user_id,
    product_id,
    count(distinct order_id)
FROM (
    SELECT
        orders.order_id,
        opt.order_id oid,
        orders.user_id,
        opt.product_id
    FROM
        orders 
    LEFT JOIN order_products_prior opt ON orders.order_id = opt.order_id
    WHERE
        opt.order_id IS NOT NULL
) AS foo
GROUP BY 1,2