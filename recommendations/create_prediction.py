from typing import List
from typing import Optional

import os
import click
import logging
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from prettytable import PrettyTable

import helpers
from create_model import Model
from config import PredictionConfig
from config import prediction_configurations


class UserItemPrediction:
    def __init__(
        self,
        config: PredictionConfig,
        input_file: Optional[str] = None,
        input_model: Optional[Model] = None,
    ) -> None:

        self.config = config
        self.persistence_path = self.config.PATHS.predictions
        self.load_model(input_file, input_model)
    
    def load_model(self, input_file: Optional[str] = None, input_model: Optional[Model] = None) -> None:
        logging.info(f'Loading model...')

        if input_file is None and input_model is None:
            self.input_file = helpers.find_latest_file(self.config.PATHS.models)
            self.model = helpers.load_input_file(self.input_file)  # type: ignore
        elif input_file is None and input_model is not None:
            self.model = input_model
        elif input_file is not None and input_model is not None:
            logging.warning('Both an input dataset and an input Dataset object were provided. Using the object.')
        elif input_file is not None and input_model is None:
            self.input_file = click.format_filename(input_file)
            self.model = helpers.load_input_file(self.input_file)  # type: ignore

    def get_lightfm_recommendation(
        self,
        user_index: int,
    ) -> List[int]:
        '''
        Main function that creates user-variant recommendation lists.
        '''
        model = self.model
        interaction_df = self.model.dataset.interaction_df
        item_df = self.model.dataset.item_df
        user_df = self.model.dataset.user_df
        item_no_interactions_list = self.model.dataset.item_no_interactions_list
        n_users, n_items = self.model.dataset.interactions.shape

        is_new_user = (user_index not in list(user_df['user_id']))

        if is_new_user:
            # TODO: Cold-start recommendation
            logging.info('Getting prediction for new user ~')
        else:
            scores = self.model.model.predict(
                user_index,
                item_ids=np.arange(n_items),
                item_features=self.model.dataset.item_features
            )
            item_df['scores'] = scores
            top_items = item_df['product_id'][np.argsort(-scores)]
            top_items_idx = top_items.index
        
        top_reccs_df = item_df.iloc[top_items_idx, :].head(10)
        topReccsTable = PrettyTable(['product_id', 'product_name', 'aisle', 'department', 'num', 'scores'])
        topReccsTable.add_row([
           top_reccs_df['product_id'],
           top_reccs_df['product_name'],
           top_reccs_df['aisle'],
           top_reccs_df['department'],
           top_reccs_df['num'],
           top_reccs_df['scores']
        ])
        logging.info(f'Top 10 Recommendations: \n{topReccsTable}')

        return top_reccs_df

    def get_similar_items(
        self,
        product_id: int,
    ) -> pd.DataFrame:
        '''
        Main function that creates similar variant recommendation lists.
        '''
        item_df = self.model.dataset.item_df

        annoy_model = AnnoyIndex(self.model.config.ANNOY_PARAMS['emb_dim'])
        annoy_model.load(self.config.PATHS.models + '/item.ann')
        similar_variants = annoy_model.get_nns_by_item(
            product_id,
            self.model.config.ANNOY_PARAMS['nn_count'],
            search_k=-1,
            include_distances=False
        )
        logging.info('inside sv')
        logging.info(type(similar_variants))
        logging.info(similar_variants)
        similar_variants_df = item_df.iloc[similar_variants, :]

        similarVariantsTable = PrettyTable(['product_id', 'product_name', 'aisle', 'department', 'num'])
        similarVariantsTable.add_row([
           similar_variants_df['product_id'],
           similar_variants_df['product_name'],
           similar_variants_df['aisle'],
           similar_variants_df['department'],
           similar_variants_df['num']
        ])
        logging.info(f'Similar Variants Data: \n{similarVariantsTable}')

        return similar_variants_df


@click.command()
@click.option('--input_file', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--user', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--config', default='production')
def main(input_file: str, config: str, user: int) -> None:

    logging.info("Let's make a prediction!")
    configuration = helpers.get_configuration(config, prediction_configurations)

    predictor = UserItemPrediction(config=configuration, input_file=None)
    predictor.get_similar_items(configuration.DEFAULT_ITEM_EG)
    predictor.get_lightfm_recommendation(configuration.DEFAULT_USER_EG)


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()