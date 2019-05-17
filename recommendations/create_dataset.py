from typing import Dict

import click
import logging
import pandas as pd
import numpy as np
import psycopg2 as pg
from lightfm.data import Dataset as LFMDataset

import helpers
from config import DataConfig
from config import data_configurations


class Dataset:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.product_data: pd.DataFrame
        self.user_data: pd.DataFrame
        self.interaction_data: pd.DataFrame

        self.persistence_path = self.config.PATHS.data

    def create(self) -> None:
        self._pull_data()
        self._preprocess_data()
        # self.build_lightfm_dataset()

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
        logging.info(f'Shape of product_details dataframe: {self.product_data.shape}')
        logging.info(f'Shape of user_details dataframe: {self.user_data.shape}')
        logging.info(f'Shape of interactions dataframe: {self.interaction_data.shape}')

        print(self.interaction_data.head())
        print(self.interaction_data.tail())
        logging.info(self.interaction_data.nunique())
        logging.info(self.product_data.nunique())
        logging.info('printing item data')
        logging.info(self.product_data.head())

        logging.info('printing user before in interaction')
        logging.info(self.user_data)
        user_data = self.user_data[
            self.user_data['user_id'].isin(self.interaction_data['user_id'])
        ].reset_index(drop=True)
        logging.info('printing user is in interaction')
        logging.info(user_data)
        logging.info(user_data.shape)

        # For use if there is no interaction users - Cold start problem
        user_data_no_interactions = self.user_data[
            ~self.user_data['user_id'].isin(self.interaction_data['user_id'])
        ].reset_index(drop=True)
        logging.info('printing no interactions')
        logging.info(user_data_no_interactions)

        product_data = self.product_data[
            self.product_data['pid'].isin(self.interaction_data['product_id'])
        ].reset_index(drop=True)

        product_data_no_interactions = self.product_data[
            ~self.product_data['pid'].isin(self.interaction_data['product_id'])
        ].reset_index(drop=True)

        interaction_products = pd.DataFrame(
            pd.unique(self.interaction_data['product_id']), columns=['product_id']
        )

        logging.info(f'Total Interaction Product Count: {len(interaction_products)}')
        logging.info(f'interaction_products Dataframe: {interaction_products.head()}')

        product_data = pd.merge(
            interaction_products, product_data, left_on='product_id', right_on='pid', how='inner'
        ).drop('pid', axis=1)

        self.user_list = user_data.to_dict(orient='records')
        self.product_list = product_data.to_dict(orient='records')
        self.interaction_list = self.interaction_data.to_dict(orient='records')

        logging.info('Printing in orient: records format.')
        logging.info(self.user_list[:10])
        logging.info(self.product_list[:10])
        logging.info(self.interaction_list[:10])

        self.user_df = user_data
        self.product_df = product_data
        self.interaction_df = self.interaction_data

        logging.info('Printing User, Product, Interaction Dataframes.')
        logging.info(self.user_df.head())
        logging.info(self.product_df.head())
        logging.info(self.interaction_df.head())
        logging.info('Printing product_data_no_interactions')
        logging.info(product_data_no_interactions)

        self.user_no_interactions_df = user_data_no_interactions
        self.product_no_interactions_df = product_data_no_interactions
        self.product_no_interactions_list = product_data_no_interactions.sort_values(['num'], ascending=False).pid.tolist()

        logging.info('printing product data no interactions')
        logging.info(product_data_no_interactions)

    def build_lightfm_dataset(self) -> None:
        logging.info("Creating LightFM matrices...")
        lightfm_dataset = LFMDataset()
        ratings_list = self.interaction_list
        logging.info('#'*60)
        lightfm_dataset.fit_partial(
            (rating['user_id'] for rating in ratings_list),
            (rating['product_id'] for rating in ratings_list)
        )

        logging.info('printing product_feature_names')
        product_feature_names = self.product_df.columns
        logging.info(product_feature_names)
        product_feature_names = product_feature_names[~product_feature_names.isin(['user_id'])]
        logging.info(product_feature_names)

        for product_feature_name in product_feature_names:
            lightfm_dataset.fit_partial(
                items=(product['product_id'] for product in self.product_list),
                item_features=((product[product_feature_name] for product in self.product_list)),
            )

        product_features_data = []
        # logging.info(self.product_list)
        for product in self.product_list:
            product_features_data.append(
                (
                    product['product_id'],
                    [
                        product['product_name'],
                        product['aisle'],
                        product['department']
                    ],
                )
            )
        logging.info(product_features_data)
        self.product_features = lightfm_dataset.build_item_features(product_features_data)
        self.interactions, self.weights = lightfm_dataset.build_interactions(
            ((rating['user_id'], rating['product_id']) for rating in ratings_list)
        )
        logging.info(self.interactions)
        logging.info(self.weights)
            

@click.command()
@click.option('--config', default='production', help='the deployment target')
def main(config: str) -> None:
    logging.info('Creating dataset.')

    configuration = helpers.get_configuration(config, data_configurations)

    dataset = Dataset(config=configuration)  # type: ignore
    dataset.create()
    # helpers.save(dataset)


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()