from pycaret.classification import *
import pandas as pd
import os

def train_classifier(model_type, data_path, timestart, save=False):
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

    print(f'Model types: {model_type}')
    print(f'Data path: {data_path}')
    
    # Load data
    data = pd.read_csv(data_path, header=0)
    data['link'] = data['relation'].apply(lambda x: 1 if x != 'no_link_known' else 0) # Convert relations to binary label
    data = data.drop(['head', 'relation', 'tail'], axis=1)

    # Experiment setup
    s = setup(data, target = 'link', fold_strategy = 'stratifiedkfold', fold=10, train_size = 0.8, n_jobs=-1, system_log=True, use_gpu = True)
    exp = ClassificationExperiment()

    # Model training
    for type in model_type:
        print(f'Model - {type} : \n {model}')
        model = create_model(type) # Train the classifier

        if save == True:
            os.makedirs(f'binary_classif/{type}', exist_ok=True)
            save_model(model, f'binary_classif/{type}/{type}_model_{timestart}')

if __name__ == '__main__':
    from datetime import datetime
    model_type = ['lr', 'lightgbm', 'rf', 'et']
    data_path = 'KGene2pheno/data/embeddings.csv'
    timestart = datetime.datetime.now()
    train_classifier(model_type, data_path, timestart, save=False)
