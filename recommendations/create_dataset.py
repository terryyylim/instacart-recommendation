from typing import Dict

import click
import logging
import psycopg2 as pg
import pandas as pd
import numpy as np

import helpers
from config import DataConfig
from config import data_configurations


class Dataset:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_data: Dict[str, pd.DataFrame] = {}

        self.persistence_path = self.config.PATHS.data

    def create(self) -> None:
        self._pull_data()
        self._preprocess_data()

    def _pull_data(self) -> None:
        for query_name, query_info in self.config.QUERIES.items():
            logging.info(f'Pulling {query_name} from database...')
            with open(query_info['file'], 'r') as f:
                sql = f.read()

            db_params: Dict[str, str] = self.config.DATABASE_CONFIG[query_info['database']]
            with pg.connect(**db_params) as connection:
                connection.autocommit = True
                result = pd.read_sql(sql, connection)
                self.raw_data[query_name] = result

    def _preprocess_data(self) -> None:
        df = self.raw_data['aisle_details']
        print(df.head())

        self.dataset = df
            

@click.command()
@click.option('--config', default='production', help='the deployment target')
def main(config: str) -> None:
    logging.info('Creating dataset.')

    configuration = helpers.get_configuration(config, data_configurations)

    dataset = Dataset(config=configuration)  # type: ignore
    dataset.create()
    helpers.save(dataset)


if __name__ == "__main__":
    main()