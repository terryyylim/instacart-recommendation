from typing import List
from typing import Optional

import logging

import os
import click
from pathlib import Path
from annoy import AnnoyIndex
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split as train_test_split

import helpers
from create_dataset import Dataset  # noqa
from config import model_configurations
from config import ModelConfig


class Model:
    def __init__(
        self,
        config: ModelConfig,
        input_file: Optional[str] = None,
        input_dataset: Optional[Dataset] = None,
    ) -> None:

        self.config = config
        self.persistence_path = self.config.PATHS.models
        self.load_dataset()

    def load_dataset(self, input_file: Optional[str] = None, input_dataset: Optional[Dataset] = None) -> None:
        logging.info(f'Loading dataset...')

        if input_file is None and input_dataset is None:
            self.input_file = helpers.find_latest_file(self.config.PATHS.data)
            self.dataset = helpers.load_input_file(self.input_file)  # type: ignore
        elif input_file is None and input_dataset is not None:
            self.dataset = input_dataset
        elif input_file is not None and input_dataset is not None:
            logging.warning('Both an input dataset and an input Dataset object were provided. Using the object.')
        elif input_file is not None and input_dataset is None:
            self.input_file = click.format_filename(input_file)
            self.dataset = helpers.load_input_file(self.input_file)  # type: ignore

    def build_model(self) -> None:
        """
        Fits model for user-variant recommendations and similar variant recommendations.
        """
        if hasattr(self, 'input_file'):
            logging.info(f'Training the main model with dataset {self.input_file}...')
        else:
            logging.info('Training the model...')

        train_validation, test = train_test_split(
            self.dataset.interactions, **self.config.VALIDATION_PARAMS
        )
        train, validation = train_test_split(
            train_validation, **self.config.VALIDATION_PARAMS
        )

        logging.info(f'train: Type; {type(train)}, Shape; {train.shape}')
        logging.info(f'validation: Type; {type(validation)}, Shape; {validation.shape}')
        logging.info(f'test: Type; {type(test)}, Shape; {test.shape}')

        model = LightFM(**self.config.LIGHTFM_PARAMS)
        warp_auc: List[float] = []
        no_improvement_rounds = 0
        best_auc = 0.0
        epochs = self.config.FIT_PARAMS['epochs']
        early_stopping_rounds = self.config.FIT_PARAMS['early_stopping_rounds']

        logging.info(
            f'Training model until validation AUC has not improved in {early_stopping_rounds} epochs...'
        )

        for epoch in range(epochs):
            logging.info(f'Epoch {epoch}...')
            if no_improvement_rounds >= early_stopping_rounds:
                break

            model.fit(
                interactions=train,
                item_features=self.dataset.item_features,
                epochs=self.config.FIT_PARAMS['epochs_per_round'],
                num_threads=self.config.FIT_PARAMS['core_count'],
            )
            warp_auc.append(
                auc_score(
                    model=model,
                    test_interactions=validation,
                    item_features=self.dataset.item_features,
                ).mean()
            )

            if warp_auc[-1] > best_auc:
                best_auc = warp_auc[-1]
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1

            logging.info(f'[{epoch}]\tvalidation_warp_auc: {warp_auc[-1]}')

        self.num_epochs = len(warp_auc) - early_stopping_rounds
        logging.info(f'Stopping. Best Iteration:')
        logging.info(
            f'[{self.num_epochs - 1}]\tvalidation_warp_auc: {warp_auc[self.num_epochs - 1]}'
        )

        logging.info(f'Calculating AUC score on test set...')
        test_score = auc_score(
            model=model,
            test_interactions=test,
            item_features=self.dataset.item_features,
        ).mean()
        logging.info(f'Test Set AUC Score: {test_score}')

        self.model = model
        self.test_score = test_score

    def build_cab_model(
        self
    ) -> None:
        '''
        Fits model for complement variant recommendations. Only interaction matrices are fed
        into the LightFM model, without any content-based information.
        '''
        if hasattr(self, 'input_file'):
            logging.info(f'Training the main CAB model with dataset {self.input_file}...')
        else:
            logging.info('Training the CAB model...')

        cab_model = LightFM(**self.config.LIGHTFM_CAB_PARAMS)
        self.cab_model = cab_model.fit(
            interactions=self.dataset.interactions,
            epochs=self.config.FIT_PARAMS['epochs']
        )

    def build_annoy_representations(
        self,
        feature_type: str,
        is_cab: bool
    ) -> None:
        '''
        Getting product/user matrix into proper representations required to
        perform Approximate Nearest Neighbors.
        ---------------
        From LightFM get_item_representations - Index 0: Item biases; Index 1: Item embeddings
        - Excluding Content-based information for CAB model since user/item features overpower MF.
        '''
        logging.info('Preparing matrix representations for ANN ~')
        if is_cab:
            file_label = '_cab.ann'
            emb_dim = self.config.ANNOY_PARAMS['cab_emb_dim']
            if feature_type == 'user':
                latent_repr_emb = self.cab_model.get_user_representations()[1]
                logging.info(
                    f'Preparing CAB Annoy object using user_features\n'
                    f'Type: {type(self.dataset.user_features)}\n'
                    f'Shape: {self.dataset.user_features.shape}'
                )
            elif feature_type == 'item':
                latent_repr_emb = self.cab_model.get_item_representations()[1]
                logging.info(
                    f'Preparing CAB Annoy object using item_features\n'
                    f'Type: {type(self.dataset.item_features)}\n'
                    f'Shape: {self.dataset.item_features.shape}'
                )
            else:
                raise ValueError('Unknown feature type passed to function')
        else:
            file_label = '.ann'
            emb_dim = self.config.ANNOY_PARAMS['emb_dim']
            if feature_type == 'user':
                latent_repr_emb = self.model.get_user_representations(
                    features=self.dataset.user_features
                )[1]
                logging.info(
                    f'Preparing Annoy object using user_features\n'
                    f'Type: {type(self.dataset.user_features)}\n'
                    f'Shape: {self.dataset.user_features.shape}'
                )
            elif feature_type == 'item':
                latent_repr_emb = self.model.get_item_representations(
                    features=self.dataset.item_features
                )[1]
                logging.info(
                    f'Preparing Annoy object using item_features\n'
                    f'Type: {type(self.dataset.item_features)}\n'
                    f'Shape: {self.dataset.item_features.shape}'
                )
            else:
                raise ValueError('Unknown feature type passed to function')

        logging.info(f'Shape of embeddings: {latent_repr_emb.shape}')
        a = AnnoyIndex(emb_dim, metric=self.config.ANNOY_PARAMS['metric'])
        for item in range(len(latent_repr_emb)):
            a.add_item(item, latent_repr_emb[item])
        a.build(self.config.ANNOY_PARAMS['trees'])

        persistence_path = Path(self.persistence_path)
        if not persistence_path.is_dir():
            persistence_path.mkdir(parents=True)

        persistance_file = os.path.join(self.persistence_path, feature_type + file_label)
        a.save(persistance_file)

        logging.info(
            f'{feature_type} vector representations saved to {persistance_file}'
        )

    def delete_var(self, classname: str, attrname: str) -> None:
        '''
        Delete unused variables that have been assigned.
        '''
        if classname == 'model':
            delattr(self, attrname)
        elif classname == 'dataset':
            delattr(self.dataset, attrname)
        else:
            raise ValueError('Invalid value passed to classname argument.  Value must be "dataset" or "model".')

    def clean_up(self) -> None:
        '''
        Delete unused repr from LightFM which exhausts available memory
        '''
        self.delete_var(classname='model', attrname='model')


@click.command()
@click.option('--input_file', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--config', default='production')
def main(input_file: str, config: str) -> None:
    logging.info("Creating model...")

    configuration = helpers.get_configuration(config, model_configurations)
    model = Model(config=configuration, input_file=input_file)

    model.build_model()
    model.build_cab_model()
    model.build_annoy_representations(feature_type='item', is_cab=True)
    model.build_annoy_representations(feature_type='item', is_cab=False)
    model.clean_up()

    try:
        helpers.save(model)
    except Exception:
        logging.info('Error while saving model.')


if __name__ == '__main__':
    logger = helpers.get_logger()

    main()
