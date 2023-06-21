from pycaret.classification import *
import pandas as pd
import os

def train_classifier(model_type, data, timestart, logger, device, save=False):
    """
    Train binary classification models on the provided data.

    Parameters
    ----------
    model_type : list
        List of estimator types to train. Must be supported by pycaret.
        See the full list at https://pycaret.readthedocs.io/en/stable/api/classification.html#pycaret.classification.create_model
    data_path : str
        The file path to the data output of embeddings.generate().
    timestart : datetime.datetime
        The starting time of the training.
    save : bool, optional
        Flag indicating whether to save the trained models (default is False).
    """

    logger.info(f'Model types: {model_type}')
    
    # Load data
    data['link'] = data['relation'].apply(lambda x: 1 if x != 'no_link_known' else 0) # Convert relations to binary label
    data = data.drop(['head', 'relation', 'tail'], axis=1)

    # Experiment setup

    logger.info(f'Experiment Setup:')
    s = setup(data, target = 'link', fold_strategy = 'stratifiedkfold', fold=10, train_size = 0.8, n_jobs=-1, system_log=logger, use_gpu=True if device == 'cuda' else False, verbose=False)
    exp = ClassificationExperiment()

    # Model training
    for type in model_type:
        logger.info(f'MODEL - {type}')
        model = create_model(type, verbose=False) # Train the classifier
        logger.info(f'CONFIG -\n {model}')
        logger.info(f'RESULTS -\n{pull()}')

        if save == True:
            os.makedirs(f'binary_classif/{type}', exist_ok=True)
            save_model(model, f'binary_classif/{type}/{type}_model_{timestart}')
    os.remove('logs.log')

if __name__ == '__main__':
    from datetime import datetime
    model_type = ['lr', 'lightgbm', 'rf', 'et']
    data_path = 'KGene2pheno/data/embeddings.csv'
    timestart = datetime.datetime.now()
    train_classifier(model_type, data_path, timestart, save=False)
