2024-03-15 08:40:14,126 INFO: Start time: 2024-03-15_08-40-14
2024-03-15 08:40:15,905 INFO: Device: cuda

2024-03-15 08:40:15.906213 - Querying celegans SPARQL endpoint with the following queries : ['molecular-entity', 'phenotype', 'interaction', 'disease_plus_ortho', 'disease-ontology', 'phenotype-ontology'].

2024-03-15 08:41:48,249 INFO: Splitting knowledge graph..
2024-03-15 08:41:53,324 INFO: Train set
2024-03-15 08:41:53,325 INFO: Number of entities: 284670
2024-03-15 08:41:53,325 INFO: Number of relation types: 7
2024-03-15 08:41:53,325 INFO: Number of triples: 447714 

2024-03-15 08:41:53,325 INFO: Test set
2024-03-15 08:41:53,325 INFO: Number of entities: 284670
2024-03-15 08:41:53,325 INFO: Number of relation types: 7
2024-03-15 08:41:53,325 INFO: Number of triples: 81676

2024-03-15 08:41:55,704 INFO: 2024-03-15 08:41:55.704487 - PARAMETERS
2024-03-15 08:41:55,704 INFO: 	 keywords : ['molecular-entity', 'phenotype', 'interaction', 'disease_plus_ortho', 'disease-ontology', 'phenotype-ontology']
2024-03-15 08:41:55,704 INFO: 	 method : ComplEx
2024-03-15 08:41:55,704 INFO: 	 dataset : celegans
2024-03-15 08:41:55,704 INFO: 	 query : None
2024-03-15 08:41:55,704 INFO: 	 normalize_parameters : True
2024-03-15 08:41:55,704 INFO: 	 train_classifier : ['rf', 'lr']
2024-03-15 08:41:55,704 INFO: 	 save_model : True
2024-03-15 08:41:55,704 INFO: 	 save_embeddings : True
2024-03-15 08:41:55,704 INFO: 	 n_epochs : 100
2024-03-15 08:41:55,704 INFO: 	 batch_size : 3072
2024-03-15 08:41:55,704 INFO: 	 lr : 0.0001
2024-03-15 08:41:55,705 INFO: 	 weight_decay : 0.0001
2024-03-15 08:41:55,705 INFO: 	 loss_fn : margin
2024-03-15 08:41:55,705 INFO: 	 ent_emb_dim : 50
2024-03-15 08:41:55,705 INFO: 	 eval_task : relation-prediction
2024-03-15 08:41:55,705 INFO: 	 split_ratio : 0.8
2024-03-15 08:41:55,705 INFO: 	 dissimilarity_type : L1
2024-03-15 08:41:55,705 INFO: 	 margin : 1.0
2024-03-15 08:41:55,705 INFO: 	 rel_emb_dim : 50
2024-03-15 08:41:55,705 INFO: 	 n_filters : 10
2024-03-15 08:41:55,705 INFO: 	 init_transe : False
2024-03-15 08:41:55,705 INFO: Training model ComplEx for 100 epochs...
2024-03-15 08:41:35.513241 - Query executed !
Function 'load_celegans' executed in 79.6071s
Function 'split' executed in 5.0746s





































































































2024-03-15 08:43:19,359 INFO: 2024-03-15 08:43:19.359291 - Finished Training of ComplEx !

2024-03-15 08:43:21,138 INFO: 2024-03-15 08:43:21.138361 - Evaluating..

2024-03-15 08:43:23,853 INFO: {'http://semanticscience.org/resource/SIO_000628': 0, 'http://semanticscience.org/resource/SIO_001279': 1, 'http://www.semanticweb.org/needed-terms#001': 2, 'http://www.semanticweb.org/needed-terms#009': 3, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 4, 'http://www.w3.org/2000/01/rdf-schema#label': 5, 'http://www.w3.org/2000/01/rdf-schema#subClassOf': 6}
2024-03-15 08:43:23,884 INFO: 
 [[    0     0     0     0     0     0     0     0]
 [ 8881 13331 37798  3211  1646     0  1451     0]
 [  272  1234  2571   937   293   966   412     0]
 [  164   443   650   575   174   482   272     0]
 [  167   365   405   465   116   294   200     0]
 [  118   266   272   407    90   231   204     0]
 [   95   191   192   298    69   184   135     0]
 [   86   239   160   250    54   216   144     0]]
2024-03-15 08:43:23,884 INFO: 2024-03-15 08:43:23.884393 - EMBEDDING MODEL EVALUATION RESULTS:
2024-03-15 08:43:23,884 INFO: Task : relation-prediction
2024-03-15 08:43:23,885 INFO: Hit@1 : 0.8119643330574036
2024-03-15 08:43:23,886 INFO: Hit@3 : 0.9276041984558105
2024-03-15 08:43:23,886 INFO: Hit@5 : 0.971680760383606
2024-03-15 08:43:23,887 INFO: Hit@10 : 1.0
2024-03-15 08:43:23,887 INFO: Mean Rank : 1.4567681550979614
2024-03-15 08:43:23,888 INFO: MRR : 0.8785842061042786
2024-03-15 08:43:23,896 INFO: Training of Embedding Model done !

2024-03-15 08:43:23,904 INFO: Converting test set to embeddings...
Function 'evaluate_emb_model' executed in 2.7505s
Function 'train' executed in 108.3835s

2024-03-15 08:43:45,180 INFO: Test set converted. It will be used to train the classifier

2024-03-15 08:43:45,181 INFO: Training classifier...
2024-03-15 08:43:45,181 INFO: Model types: ['rf', 'lr']
2024-03-15 08:43:45,273 INFO: Experiment Setup:
2024-03-15 08:43:45,275 INFO: PyCaret ClassificationExperiment
2024-03-15 08:43:45,275 INFO: Logging name: clf-default-name
2024-03-15 08:43:45,275 INFO: ML Usecase: MLUsecase.CLASSIFICATION
2024-03-15 08:43:45,275 INFO: version 3.2.0
2024-03-15 08:43:45,275 INFO: Initializing setup()
2024-03-15 08:43:45,275 INFO: self.USI: 18fb
2024-03-15 08:43:45,275 INFO: self._variable_keys: {'is_multiclass', 'fold_shuffle_param', 'log_plots_param', 'data', 'fix_imbalance', 'gpu_n_jobs_param', 'y_train', 'exp_name_log', 'logging_param', 'fold_generator', 'y', 'idx', 'fold_groups_param', 'pipeline', 'memory', 'exp_id', '_ml_usecase', 'X_test', 'USI', 'n_jobs_param', 'seed', 'X_train', 'X', 'html_param', 'target_param', 'gpu_param', 'y_test', '_available_plots'}
2024-03-15 08:43:45,275 INFO: Checking environment
2024-03-15 08:43:45,275 INFO: python_version: 3.10.13
2024-03-15 08:43:45,275 INFO: python_build: ('main', 'Sep 11 2023 13:44:35')
2024-03-15 08:43:45,275 INFO: machine: x86_64
2024-03-15 08:43:45,277 INFO: platform: Linux-6.1.0-18-amd64-x86_64-with-glibc2.36
2024-03-15 08:43:45,278 INFO: Memory: svmem(total=270124969984, available=249209864192, percent=7.7, used=18676264960, free=23859519488, active=75917053952, inactive=160585441280, buffers=3597434880, cached=223991750656, shared=333881344, slab=7817924608)
2024-03-15 08:43:45,279 INFO: Physical Core: 32
2024-03-15 08:43:45,279 INFO: Logical Core: 64
2024-03-15 08:43:45,279 INFO: Checking libraries
2024-03-15 08:43:45,279 INFO: System:
2024-03-15 08:43:45,279 INFO:     python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]
2024-03-15 08:43:45,279 INFO: executable: /home/heligon/anaconda3/envs/Antoine/bin/python
2024-03-15 08:43:45,279 INFO:    machine: Linux-6.1.0-18-amd64-x86_64-with-glibc2.36
2024-03-15 08:43:45,279 INFO: PyCaret required dependencies:
2024-03-15 08:43:45,300 INFO:                  pip: 23.3.1
2024-03-15 08:43:45,300 INFO:           setuptools: 68.2.2
2024-03-15 08:43:45,300 INFO:              pycaret: 3.2.0
2024-03-15 08:43:45,300 INFO:              IPython: 8.21.0
2024-03-15 08:43:45,300 INFO:           ipywidgets: 8.1.1
2024-03-15 08:43:45,300 INFO:                 tqdm: 4.66.1
2024-03-15 08:43:45,300 INFO:                numpy: 1.25.2
2024-03-15 08:43:45,300 INFO:               pandas: 1.5.3
2024-03-15 08:43:45,300 INFO:               jinja2: 3.1.3
2024-03-15 08:43:45,300 INFO:                scipy: 1.10.1
2024-03-15 08:43:45,300 INFO:               joblib: 1.3.2
2024-03-15 08:43:45,300 INFO:              sklearn: 1.2.2
2024-03-15 08:43:45,300 INFO:                 pyod: 1.1.2
2024-03-15 08:43:45,300 INFO:             imblearn: 0.12.0
2024-03-15 08:43:45,300 INFO:    category_encoders: 2.6.3
2024-03-15 08:43:45,300 INFO:             lightgbm: 4.3.0
2024-03-15 08:43:45,300 INFO:                numba: 0.59.0
2024-03-15 08:43:45,300 INFO:             requests: 2.31.0
2024-03-15 08:43:45,300 INFO:           matplotlib: 3.6.0
2024-03-15 08:43:45,300 INFO:           scikitplot: 0.3.7
2024-03-15 08:43:45,300 INFO:          yellowbrick: 1.5
2024-03-15 08:43:45,300 INFO:               plotly: 5.18.0
2024-03-15 08:43:45,300 INFO:     plotly-resampler: Not installed
2024-03-15 08:43:45,300 INFO:              kaleido: 0.2.1
2024-03-15 08:43:45,300 INFO:            schemdraw: 0.15
2024-03-15 08:43:45,300 INFO:          statsmodels: 0.14.1
2024-03-15 08:43:45,300 INFO:               sktime: 0.21.1
2024-03-15 08:43:45,300 INFO:                tbats: 1.1.3
2024-03-15 08:43:45,300 INFO:             pmdarima: 2.0.4
2024-03-15 08:43:45,300 INFO:               psutil: 5.9.8
2024-03-15 08:43:45,301 INFO:           markupsafe: 2.1.5
2024-03-15 08:43:45,301 INFO:              pickle5: Not installed
2024-03-15 08:43:45,301 INFO:          cloudpickle: 3.0.0
2024-03-15 08:43:45,301 INFO:          deprecation: 2.1.0
2024-03-15 08:43:45,301 INFO:               xxhash: 3.4.1
2024-03-15 08:43:45,301 INFO:            wurlitzer: 3.0.3
2024-03-15 08:43:45,301 INFO: PyCaret optional dependencies:
2024-03-15 08:43:45,324 INFO:                 shap: Not installed
2024-03-15 08:43:45,324 INFO:            interpret: Not installed
2024-03-15 08:43:45,324 INFO:                 umap: Not installed
2024-03-15 08:43:45,325 INFO:      ydata_profiling: Not installed
2024-03-15 08:43:45,325 INFO:   explainerdashboard: Not installed
2024-03-15 08:43:45,325 INFO:              autoviz: Not installed
2024-03-15 08:43:45,325 INFO:            fairlearn: Not installed
2024-03-15 08:43:45,325 INFO:           deepchecks: Not installed
2024-03-15 08:43:45,325 INFO:              xgboost: Not installed
2024-03-15 08:43:45,325 INFO:             catboost: Not installed
2024-03-15 08:43:45,325 INFO:               kmodes: Not installed
2024-03-15 08:43:45,325 INFO:              mlxtend: Not installed
2024-03-15 08:43:45,325 INFO:        statsforecast: Not installed
2024-03-15 08:43:45,325 INFO:         tune_sklearn: Not installed
2024-03-15 08:43:45,325 INFO:                  ray: Not installed
2024-03-15 08:43:45,325 INFO:             hyperopt: Not installed
2024-03-15 08:43:45,325 INFO:               optuna: Not installed
2024-03-15 08:43:45,325 INFO:                skopt: Not installed
2024-03-15 08:43:45,325 INFO:               mlflow: Not installed
2024-03-15 08:43:45,325 INFO:               gradio: Not installed
2024-03-15 08:43:45,325 INFO:              fastapi: Not installed
2024-03-15 08:43:45,325 INFO:              uvicorn: Not installed
2024-03-15 08:43:45,325 INFO:               m2cgen: Not installed
2024-03-15 08:43:45,325 INFO:            evidently: Not installed
2024-03-15 08:43:45,325 INFO:                fugue: Not installed
2024-03-15 08:43:45,325 INFO:            streamlit: Not installed
2024-03-15 08:43:45,325 INFO:              prophet: Not installed
2024-03-15 08:43:45,325 INFO: None
2024-03-15 08:43:45,325 INFO: Set up data.
2024-03-15 08:43:45,622 INFO: Set up folding strategy.
2024-03-15 08:43:45,622 INFO: Set up train/test split.
2024-03-15 08:43:45,987 INFO: Set up index.
2024-03-15 08:43:45,990 INFO: Assigning column types.
2024-03-15 08:43:46,271 INFO: Engine successfully changes for model 'lr' to 'sklearn'.
2024-03-15 08:43:46,303 INFO: Engine for model 'knn' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,305 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,360 INFO: Engine for model 'knn' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,360 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,380 INFO: Engine successfully changes for model 'knn' to 'sklearn'.
2024-03-15 08:43:46,413 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,465 INFO: Engine for model 'rbfsvm' has not been set explicitly, hence returning None.
2024-03-15 08:43:46,486 INFO: Engine successfully changes for model 'rbfsvm' to 'sklearn'.
2024-03-15 08:43:46,592 INFO: Preparing preprocessing pipeline...
2024-03-15 08:43:46,635 INFO: Set up simple imputation.
2024-03-15 08:43:47,414 INFO: Finished creating preprocessing pipeline.
2024-03-15 08:43:47,418 INFO: Pipeline: Pipeline(memory=FastMemory(location=/tmp/joblib),
         steps=[('numerical_imputer',
                 TransformerWrapper(exclude=None,
                                    include=['0', '1', '2', '3', '4', '5', '6',
                                             '7', '8', '9', '10', '11', '12',
                                             '13', '14', '15', '16', '17', '18',
                                             '19', '20', '21', '22', '23', '24',
                                             '25', '26', '27', '28', '29', ...],
                                    transformer=SimpleImputer(add_indicator=False,
                                                              copy=True,
                                                              fill_value=None,
                                                              keep_empty_features=False,
                                                              missing_values=nan,
                                                              strategy='mean',
                                                              verbose='deprecated'))),
                ('categorical_imputer',
                 TransformerWrapper(exclude=None, include=[],
                                    transformer=SimpleImputer(add_indicator=False,
                                                              copy=True,
                                                              fill_value=None,
                                                              keep_empty_features=False,
                                                              missing_values=nan,
                                                              strategy='most_frequent',
                                                              verbose='deprecated')))],
         verbose=False)
2024-03-15 08:43:47,418 INFO: Creating final display dataframe.
2024-03-15 08:43:48,547 INFO: Setup _display_container:                     Description             Value
0                    Session id              5353
1                        Target              link
2                   Target type            Binary
3           Original data shape     (163352, 201)
4        Transformed data shape     (163352, 201)
5   Transformed train set shape     (130681, 201)
6    Transformed test set shape      (32671, 201)
7              Numeric features               200
8                    Preprocess              True
9               Imputation type            simple
10           Numeric imputation              mean
11       Categorical imputation              mode
12               Fold Generator   StratifiedKFold
13                  Fold Number                10
14                     CPU Jobs                -1
15                      Use GPU             False
16               Log Experiment             False
17              Experiment Name  clf-default-name
18                          USI              18fb
2024-03-15 08:43:48,655 INFO: setup() successfully completed in 3.38s...............
2024-03-15 08:43:48,656 INFO: MODEL - rf
2024-03-15 08:43:48,656 INFO: Initializing create_model()
2024-03-15 08:43:48,656 INFO: create_model(self=<pycaret.classification.oop.ClassificationExperiment object at 0x7fdea3d037f0>, estimator=rf, fold=None, round=4, cross_validation=True, predict=True, fit_kwargs=None, groups=None, refit=True, probability_threshold=None, experiment_custom_tags=None, verbose=False, system=True, add_to_model_list=True, metrics=None, display=None, model_only=True, return_train_score=False, error_score=0.0, kwargs={})
2024-03-15 08:43:48,656 INFO: Checking exceptions
2024-03-15 08:43:48,657 INFO: Importing libraries
2024-03-15 08:43:48,657 INFO: Copying training dataset
2024-03-15 08:43:48,971 INFO: Defining folds
2024-03-15 08:43:48,972 INFO: Declaring metric variables
2024-03-15 08:43:48,972 INFO: Importing untrained model
2024-03-15 08:43:48,972 INFO: Random Forest Classifier Imported successfully
2024-03-15 08:43:48,972 INFO: Starting cross validation
2024-03-15 08:43:48,973 INFO: Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
2024-03-15 08:45:12,459 INFO: Calculating mean and std
2024-03-15 08:45:12,460 INFO: Creating metrics dataframe
2024-03-15 08:45:12,466 INFO: Finalizing model
2024-03-15 08:45:24,115 INFO: Uploading results into container
2024-03-15 08:45:24,117 INFO: Uploading model into container now
2024-03-15 08:45:24,118 INFO: _master_model_container: 1
2024-03-15 08:45:24,118 INFO: _display_container: 2
2024-03-15 08:45:24,119 INFO: RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=5353, verbose=0, warm_start=False)
2024-03-15 08:45:24,119 INFO: create_model() successfully completed......................................
2024-03-15 08:45:24,773 INFO: CONFIG -
 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=5353, verbose=0, warm_start=False)
2024-03-15 08:45:24,780 INFO: RESULTS -
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold                                                          
0       0.9410  0.9815  0.9734  0.9142  0.9429  0.8820  0.8839
1       0.9394  0.9810  0.9737  0.9112  0.9414  0.8788  0.8809
2       0.9430  0.9813  0.9729  0.9180  0.9446  0.8860  0.8876
3       0.9372  0.9804  0.9743  0.9070  0.9394  0.8743  0.8768
4       0.9420  0.9813  0.9769  0.9132  0.9440  0.8840  0.8862
5       0.9377  0.9809  0.9737  0.9083  0.9399  0.8754  0.8777
6       0.9415  0.9825  0.9760  0.9130  0.9434  0.8829  0.8850
7       0.9429  0.9814  0.9752  0.9160  0.9447  0.8858  0.8877
8       0.9409  0.9820  0.9735  0.9139  0.9428  0.8818  0.8837
9       0.9445  0.9818  0.9784  0.9163  0.9463  0.8890  0.8911
Mean    0.9410  0.9814  0.9748  0.9131  0.9429  0.8820  0.8840
Std     0.0022  0.0005  0.0017  0.0033  0.0021  0.0044  0.0043
2024-03-15 08:45:25,359 INFO: MODEL - lr
2024-03-15 08:45:25,359 INFO: Initializing create_model()
2024-03-15 08:45:25,359 INFO: create_model(self=<pycaret.classification.oop.ClassificationExperiment object at 0x7fdea3d037f0>, estimator=lr, fold=None, round=4, cross_validation=True, predict=True, fit_kwargs=None, groups=None, refit=True, probability_threshold=None, experiment_custom_tags=None, verbose=False, system=True, add_to_model_list=True, metrics=None, display=None, model_only=True, return_train_score=False, error_score=0.0, kwargs={})
2024-03-15 08:45:25,359 INFO: Checking exceptions
2024-03-15 08:45:25,360 INFO: Importing libraries
2024-03-15 08:45:25,360 INFO: Copying training dataset
2024-03-15 08:45:25,685 INFO: Defining folds
2024-03-15 08:45:25,685 INFO: Declaring metric variables
2024-03-15 08:45:25,686 INFO: Importing untrained model
2024-03-15 08:45:25,686 INFO: Logistic Regression Imported successfully
2024-03-15 08:45:25,686 INFO: Starting cross validation
2024-03-15 08:45:25,687 INFO: Cross validating with StratifiedKFold(n_splits=10, random_state=None, shuffle=False), n_jobs=-1
2024-03-15 08:45:33,399 INFO: Calculating mean and std
2024-03-15 08:45:33,400 INFO: Creating metrics dataframe
2024-03-15 08:45:33,405 INFO: Finalizing model
2024-03-15 08:45:36,042 INFO: Uploading results into container
2024-03-15 08:45:36,043 INFO: Uploading model into container now
2024-03-15 08:45:36,044 INFO: _master_model_container: 2
2024-03-15 08:45:36,044 INFO: _display_container: 3
2024-03-15 08:45:36,044 INFO: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5353, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
2024-03-15 08:45:36,044 INFO: create_model() successfully completed......................................
2024-03-15 08:45:36,703 INFO: CONFIG -
 LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5353, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
2024-03-15 08:45:36,710 INFO: RESULTS -
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold                                                          
0       0.8995  0.9442  0.9273  0.8785  0.9022  0.7991  0.8003
1       0.8972  0.9454  0.9256  0.8758  0.9000  0.7943  0.7956
2       0.9025  0.9489  0.9265  0.8841  0.9048  0.8050  0.8060
3       0.9013  0.9442  0.9328  0.8775  0.9043  0.8026  0.8042
4       0.9014  0.9443  0.9256  0.8829  0.9038  0.8029  0.8038
5       0.8990  0.9439  0.9270  0.8778  0.9017  0.7980  0.7992
6       0.9005  0.9462  0.9264  0.8808  0.9030  0.8010  0.8021
7       0.9032  0.9482  0.9291  0.8833  0.9056  0.8064  0.8075
8       0.9011  0.9463  0.9264  0.8818  0.9036  0.8023  0.8033
9       0.9058  0.9504  0.9302  0.8869  0.9080  0.8116  0.8126
Mean    0.9012  0.9462  0.9277  0.8809  0.9037  0.8023  0.8035
Std     0.0023  0.0022  0.0022  0.0033  0.0021  0.0045  0.0044
2024-03-15 08:45:37,275 INFO: Classifier trained !

2024-03-15 08:45:37,276 INFO: Saving embeddings...
Function 'generate' executed in 21.2764s
Transformation Pipeline and Model Successfully Saved
Transformation Pipeline and Model Successfully Saved

Function 'generate' executed in 121.6576s

Function 'generate' executed in 23.3857s