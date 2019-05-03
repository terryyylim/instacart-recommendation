import click
import logging
import psycopg2
import pandas as pd

import config
import helpers
import schema

@click.command()
@click.option('--file', default='production', help='the dataset to load to Postgres database')
def main(file: str) -> None:
    logging.info(f'Loading {file} into Postgres database.')

    # Schema Details
    schema_info = schema.DATA_SCHEMA[file]
    table_name = schema_info['tablename']
    file_name = schema_info['filename']
    query = schema_info['query']

    # Clean data
    if file == 'products':
        temp_df = pd.read_csv(file_name)
        df = helpers.clean_data(temp_df)
        df.to_csv(f'tempdata/{file}.csv', index=False)
        file_name = f'tempdata/{file}.csv'

    logging.info(f'Inserting data into {table_name} table.')
    conn = psycopg2.connect(f'host={config.CREDENTIALS['host']} dbname={config.CREDENTIALS['database']} user={config.CREDENTIALS['user']}')
    logging.info(conn)

    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    with open(file_name, 'r') as f:
        next(f)
        cur.copy_from(f, table_name, sep=',')
    
    conn.commit()


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()