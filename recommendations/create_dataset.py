from typing import Dict

import click
import logging
import pandas as pd
import numpy as np
import psycopg2 as pg
from prettytable import PrettyTable
from lightfm.data import Dataset as LFMDataset

import helpers
from config import DataConfig
from config import data_configurations


class Dataset:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.item_data: pd.DataFrame
        self.user_data: pd.DataFrame
        self.interaction_data: pd.DataFrame

        self.persistence_path = self.config.PATHS.data

    def create(self) -> None:
        self._pull_data()
        self._preprocess_data()
        self.build_lightfm_dataset()

    def _pull_data(self) -> None:
        for query_name, query_info in self.config.QUERIES.items():
            logging.info(f'Pulling {query_name} from database...')
            with open(query_info['file'], 'r') as f:
                sql = f.read()

            db_params: Dict[str, str] = self.config.DATABASE_CONFIG[query_info['database']]
            with pg.connect(**db_params) as connection:
                connection.autocommit = True
                result = pd.read_sql(sql, connection)
                setattr(self, query_name, result)

    def _preprocess_data(self) -> None:
        logging.info(f'Shape of item_data dataframe: {self.item_data.shape}')
        logging.info(f'Shape of user_data dataframe: {self.user_data.shape}')
        logging.info(f'Shape of interactions dataframe: {self.interaction_data.shape}')

        logging.info(f'Interaction Data (head): \n{self.interaction_data.head()}')
        logging.info(f'Interaction Data (tail): \n{self.interaction_data.tail()}')
        logging.info(f'Unique Counts (Interaction Data) \n{self.interaction_data.nunique()}')
        logging.info(f'Unique Counts (Item Data) \n{self.item_data.nunique()}')

        itemTable = PrettyTable(['pid', 'product_name', 'aisle', 'department', 'num'])
        itemTable.add_row([
           self.item_data['pid'],
           self.item_data['product_name'],
           self.item_data['aisle'],
           self.item_data['department'],
           self.item_data['num']
        ])
        logging.info(f'Item Data: \n{itemTable}')

        logging.info('printing user before in interaction')
        logging.info(self.user_data)
        user_data = self.user_data[
            self.user_data['user_id'].isin(self.interaction_data['user_id'])
        ].reset_index(drop=True)
        logging.info('Logging Users with interaction')
        logging.info(user_data)
        logging.info(user_data.shape)

        # For use if there is no interaction users - Cold start problem
        user_data_no_interactions = self.user_data[
            ~self.user_data['user_id'].isin(self.interaction_data['user_id'])
        ].reset_index(drop=True)
        logging.info('Logging Users with no interaction')
        logging.info(user_data_no_interactions)

        item_data = self.item_data[
            self.item_data['pid'].isin(self.interaction_data['product_id'])
        ].reset_index(drop=True)

        item_data_no_interactions = self.item_data[
            ~self.item_data['pid'].isin(self.interaction_data['product_id'])
        ].reset_index(drop=True)
        logging.info('Logging Items with no interaction')
        logging.info(f'\n{item_data_no_interactions}')

        interaction_variants = pd.DataFrame(
            pd.unique(self.interaction_data['product_id']), columns=['product_id']
        )

        logging.info(f'Total Interaction Variant Count: {len(interaction_variants)}')
        logging.info(f'Interaction_variants Dataframe: \n{interaction_variants.head()}')

        item_data = pd.merge(
            interaction_variants, item_data, left_on='product_id', right_on='pid', how='inner'
        ).drop('pid', axis=1)

        self.user_list = user_data.to_dict(orient='records')
        self.item_list = item_data.to_dict(orient='records')
        self.interaction_list = self.interaction_data.to_dict(orient='records')

        logging.info('Logging in orient: records format.')
        logging.info(f'User List (first 10 items): \n{self.user_list[:10]}')
        logging.info(f'Item List (first 10 items): \n{self.item_list[:10]}')
        logging.info(f'Interaction List (first 10 items): \n{self.interaction_list[:10]}')

        self.user_df = user_data
        self.item_df = item_data
        self.interaction_df = self.interaction_data

        logging.info('Printing User, Product, Interaction Dataframes.')
        logging.info(f'User DataFrame: \n{self.user_df.head()}')
        logging.info(f'Item DataFrame: \n{self.item_df.head()}')
        logging.info(f'Interaction DataFrame: \n{self.interaction_df.head()}')

        self.user_no_interactions_df = user_data_no_interactions
        self.item_no_interactions_df = item_data_no_interactions
        self.item_no_interactions_list = item_data_no_interactions.sort_values(['num'], ascending=False).pid.tolist()

        logging.info('Logging no interaction list')
        logging.info(f'No Interaction List: \n{self.item_no_interactions_list}')

    def build_lightfm_dataset(self) -> None:
        """
        Builds final datasets for user-variant and variant-variant recommendations.
        """
        logging.info("Creating LightFM matrices...")
        lightfm_dataset = LFMDataset()
        ratings_list = self.interaction_list
        logging.info('#'*60)
        lightfm_dataset.fit_partial(
            (rating['user_id'] for rating in ratings_list),
            (rating['product_id'] for rating in ratings_list)
        )

        item_feature_names = self.item_df.columns
        logging.info(f'Logging item_feature_names - with product_id: \n{item_feature_names}')
        item_feature_names = item_feature_names[~item_feature_names.isin(['product_id'])]
        logging.info(f'Logging item_feature_names - without product_id: \n{item_feature_names}')

        for item_feature_name in item_feature_names:
            lightfm_dataset.fit_partial(
                items=(item['product_id'] for item in self.item_list),
                item_features=((item[item_feature_name] for item in self.item_list)),
            )

        item_features_data = []
        for item in self.item_list:
            item_features_data.append(
                (
                    item['product_id'],
                    [
                        item['product_name'],
                        item['aisle'],
                        item['department']
                    ],
                )
            )
        logging.info(f'Logging item_features_data @build_lightfm_dataset: \n{item_features_data}')
        self.item_features = lightfm_dataset.build_item_features(item_features_data)
        self.interactions, self.weights = lightfm_dataset.build_interactions(
            ((rating['user_id'], rating['product_id']) for rating in ratings_list)
        )

        self.n_users, self.n_items = self.interactions.shape

        logging.info(f'Logging self.interactions @build_lightfm_dataset: \n{self.interactions}')
        logging.info(f'Logging self.weights @build_lightfm_dataset: \n{self.weights}')
        logging.info(
            f'The shape of self.interactions {self.interactions.shape} '
            f'and self.weights {self.weights.shape} represent the user-item matrix.')
            

@click.command()
@click.option('--config', default='production', help='the deployment target')
def main(config: str) -> None:
    logging.info('Creating dataset.')

    configuration = helpers.get_configuration(config, data_configurations)

    dataset = Dataset(config=configuration)  # type: ignore
    dataset.create()
    helpers.save(dataset)


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()