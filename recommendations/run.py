import click
import logging

import helpers
from config import data_configurations
from config import model_configurations
from config import prediction_configurations
from create_dataset import Dataset  # NOQA
from create_model import Model  # NOQA
from create_prediction import UserItemPrediction  # NOQA


@click.command()
@click.option('--config', default='production', help='the deployment target')
def main(config: str) -> None:

    if config not in ('production'):
        raise ValueError(f'Unknown deployment environment "{config}"')

    try:
        # Dataset
        logging.info("Creating dataset...")
        data_configuration = helpers.get_configuration(config, data_configurations)
        dataset = Dataset(config=data_configuration)
        dataset.create()

        # Model
        logging.info("Creating model...")
        model_configuration = helpers.get_configuration(config, model_configurations)
        model = Model(model_configuration, input_dataset=dataset)
        model.build_model()
        model.build_annoy_representations('item')

        # Prediction
        logging.info("Creating predictions...")
        prediction_configuration = helpers.get_configuration(config, prediction_configurations)
        predictor = UserItemPrediction(config=prediction_configuration)
        predictor.get_similar_items(prediction_configuration.DEFAULT_ITEM_EG)
        predictor.get_lightfm_recommendation(prediction_configuration.DEFAULT_USER_EG)

    except Exception as e:
        logging.exception(e)
    else:
        logging.info('Success @run.py')


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()
