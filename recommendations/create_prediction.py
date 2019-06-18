from typing import List
from typing import Optional

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
        self.user_df = self.model.dataset.user_df
        self.item_df = self.model.dataset.item_df
        self.interaction_df = self.model.dataset.interaction_df
        self.item_no_interactions_list = self.model.dataset.item_no_interactions_list

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

    def create_scores_matrix(
        self,
        is_cab: bool
    ) -> None:
        '''
        Formula: r^ui = f(q^u . p^i + b^u + b^i)
        -------
        r^ui = Prediction for user u and item i
        q^u = User representations
        p^i = Item representations
        b^u = User feature bias
        b^i = Item feature bias
        '''
        if is_cab:
            model = self.model.cab_model
        else:
            model = self.model.model
        user_bias, user_latent_repr = model.get_user_representations()
        item_bias, item_latent_repr = model.get_item_representations()
        logging.info(
            f'Logging user latent features before broadcasting\n'
            f'Type: {type(user_latent_repr)}\n'
            f'Shape: {user_latent_repr.shape}'
        )
        logging.info(
            f'Logging item latent features before broadcasting\n'
            f'Type: {type(item_latent_repr)}\n'
            f'Shape: {item_latent_repr.shape}'
        )
        logging.info(
            f'Logging user bias features before broadcasting\n'
            f'Type: {type(user_bias)}\n'
            f'Shape: {user_bias.shape}'
        )
        logging.info(
            f'Logging item bias features before broadcasting\n'
            f'Type: {type(item_bias)}\n'
            f'Shape: {item_bias.shape}'
        )

        user_bias = user_bias[:, np.newaxis]
        item_bias = item_bias[:, np.newaxis]

        self.dot_product = user_latent_repr @ item_latent_repr.T + user_bias + item_bias.T

    def get_lightfm_recommendation(
        self,
        user_index: int,
        use_precomputed_scores: bool
    ) -> List[int]:
        '''
        Top-picks
        ---------
        Main function that creates user-variant recommendation lists.
        '''
        n_users, n_items = self.model.dataset.interactions.shape

        is_new_user = (user_index not in list(self.user_df['user_id']))
        logging.info('logging in create_prediction file')
        logging.info('logging all users')
        logging.info(self.user_df)
        logging.info(self.user_df['user_id'])
        logging.info(is_new_user)
        logging.info(self.model.dataset)
        logging.info(type(self.model.dataset.item_features))
        logging.info(self.model.dataset.item_features)
        logging.info(self.model.dataset.item_features.shape)

        if is_new_user:
            # TODO: Cold-start recommendation
            logging.info('Getting prediction for new user ~')
        else:
            if use_precomputed_scores:
                scores = self.dot_product[user_index]
            else:
                scores = self.model.model.predict(
                    user_index,
                    item_ids=np.arange(n_items),
                    item_features=self.model.dataset.item_features
                )
            self.item_df['scores'] = scores
            top_items = self.item_df['product_id'][np.argsort(-scores)]
            top_items_idx = top_items.index

        top_reccs_df = self.item_df.iloc[top_items_idx, :].head(10)
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
        rec_type: int
    ) -> pd.DataFrame:
        '''
        Function that creates recommendation lists.

        The intuition behind using less components is reducing the number of latent factors
        that can be inferred. And, by excluding item features for the CAB model, recommendations
        will be less based off explicit features such as `aisle` and `department`.
        -------------------
        type:
        1 - Similar Items [DEFAULT_PARAMS]
        2 - Complement Items [CAB_PARAMS]
        '''
        logging.info(f'Logging recommendations for {self.model.config.ANNOY_PARAMS[rec_type]}')
        if rec_type == 1:
            annoy_model = AnnoyIndex(self.model.config.LIGHTFM_PARAMS['no_components'])
            annoy_model.load(self.config.PATHS.models + '/item.ann')
        elif rec_type == 2:
            annoy_model = AnnoyIndex(self.model.config.LIGHTFM_CAB_PARAMS['no_components'])
            annoy_model.load(self.config.PATHS.models + '/item_cab.ann')
        similar_variants = annoy_model.get_nns_by_item(
            product_id,
            self.model.config.ANNOY_PARAMS['nn_count'],
            search_k=-1,
            include_distances=False
        )

        logging.info(type(similar_variants))
        logging.info(similar_variants)
        similar_variants_df = self.item_df.iloc[similar_variants, :]

        similarVariantsTable = PrettyTable(['product_id', 'product_name', 'aisle', 'department', 'num'])
        similarVariantsTable.add_row([
           similar_variants_df['product_id'],
           similar_variants_df['product_name'],
           similar_variants_df['aisle'],
           similar_variants_df['department'],
           similar_variants_df['num']
        ])
        logging.info(f'{self.model.config.ANNOY_PARAMS[rec_type]} Data: \n{similarVariantsTable}')

        return similar_variants_df

    def cache_top_picks(self) -> None:
        '''
        Function to store top recommendations for each user into a dictionary.
        '''
        logging.info('Getting Top-Picks for each User')
        user_toppicks_cache = {}
        for uid in range(len(self.user_df)):
            logging.info(f'Caching Top-Picks Recommendation for User {uid}')
            user_toppicks_cache[uid] = self.get_lightfm_recommendation(user_index=uid, use_precomputed_scores=False)

        self.user_toppicks_cache = user_toppicks_cache


@click.command()
@click.option('--input_file', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--config', default='production')
def main(input_file: str, config: str) -> None:

    logging.info("Let's make a prediction!")
    configuration = helpers.get_configuration(config, prediction_configurations)

    predictor = UserItemPrediction(config=configuration, input_file=None)
    # predictor.create_scores_matrix(is_cab=False)
    # predictor.create_scores_matrix(is_cab=True)
    predictor.get_similar_items(product_id=configuration.DEFAULT_ITEM_EG, rec_type=1)
    predictor.get_similar_items(product_id=configuration.DEFAULT_ITEM_EG, rec_type=2)
    predictor.get_lightfm_recommendation(user_index=configuration.DEFAULT_USER_EG, use_precomputed_scores=False)


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()
