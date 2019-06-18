import click
import logging
import psycopg2

import helpers
import schema


@click.command()
@click.option('--file', default='production', help='the dataset to drop from Postgres database')
def main(file: str) -> None:
    logging.info(f'Loading {file} into Postgres database.')

    # Schema Details
    schema_info = schema.DATA_SCHEMA[file]
    table_name = schema_info['tablename']
    dropquery = schema_info['dropquery']

    logging.info(f'Dropping {table_name} table.')
    conn = psycopg2.connect("host=localhost dbname=postgres user=postgres")
    logging.info(conn)

    cur = conn.cursor()
    cur.execute(dropquery)
    conn.commit()


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()
