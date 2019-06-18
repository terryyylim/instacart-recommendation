DATA_SCHEMA = {
    'aisles': {
        'filename': 'data/aisles.csv',
        'tablename': 'aisles',
        'query': """
                CREATE TABLE aisles(
                    aisle_id integer PRIMARY KEY,
                    aisle text
                )
            """,
        'dropquery': """DROP TABLE aisles"""
    },
    'departments': {
        'filename': 'data/departments.csv',
        'tablename': 'departments',
        'query': """
                CREATE TABLE departments(
                    department_id integer PRIMARY KEY,
                    department text
                )
            """,
        'dropquery': """DROP TABLE departments"""
    },
    'order_products_prior': {
        'filename': 'data/order_products__prior.csv',
        'tablename': 'order_products_prior',
        'query': """
                CREATE TABLE order_products_prior(
                    order_id integer,
                    product_id integer,
                    add_to_cart integer,
                    reordered integer
                )
            """,
        'dropquery': """DROP TABLE order_products_prior"""
    },
    'order_products_train': {
        'filename': 'data/order_products__train.csv',
        'tablename': 'order_products_train',
        'query': """
                CREATE TABLE order_products_train(
                    order_id integer,
                    product_id integer,
                    add_to_cart integer,
                    reordered integer
                )
            """,
        'dropquery': """DROP TABLE order_products_train"""
    },
    'products': {
        'filename': 'data/products.csv',
        'tablename': 'products',
        'query': """
                CREATE TABLE products(
                    product_id integer PRIMARY KEY,
                    product_name text,
                    aisle_id integer,
                    department_id integer
                )
            """,
        'dropquery': """DROP TABLE products"""
    },
    'orders': {
        'filename': 'data/orders.csv',
        'tablename': 'orders',
        'query': """
                CREATE TABLE orders(
                    order_id integer PRIMARY KEY,
                    user_id integer,
                    eval_set text,
                    order_number integer,
                    order_dow integer,
                    order_hour_of_day integer,
                    days_since_prior text
                )
            """,
        'dropquery': """DROP TABLE orders"""
    }
}
