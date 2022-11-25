# author: Nikita Susan Easow
# date: 2022-11-24

"""
Takes the preprocessed training and test data and does cross validation with logistic regression and svm classifier - finds logistic regression to be the better model and does hyperparameter tuning to get the best hyperparameters. It then fits this trained model on the unseen data (test dataset).
   
Usage: fit_new_york_grading_prediction_model.py --train_data=<train_input_file> --test_data=<test_input_file> --output_dir=<output_directory>
Options:
--train_data=<train_input_file>       Path of the input file that contains the train data
--test_data=<test_input_file>         Path of the input file that contains the test data
--output_dir=<output_directory>       Path of the output file where results of the analysis will be stored 
Command to run the script:
python src/fit_new_york_grading_prediction_model.py --train_data="./data/processed/train_df.csv" --test_data="./data/processed/test_df.csv" --output_dir="./results/"
"""

from docopt import docopt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from scipy.stats import randint
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
import pickle

opt = docopt(__doc__)

def main(train_data_file, test_data_file, output_dir_path):
    """
    Takes the preprocessed training and test data and does cross validation with logistic regression and svm classifier - finds logistic regression to be the better model and does hyperparameter tuning to get the best hyperparameters. It then fits this trained model on the unseen data (test dataset).
    
    Parameters
    ----------
    string : train_data_file
        Relative path of the file that contains the preprocessed training data
    string : test_data_file
        Relative path of the file which contains the preprocessed testing data
    string : output_dir_path
        Relative path of the directory where results will be share
    
    """ 
    # read train and test data from csv files

    print("Reading data from CSV files...")
    train_df = pd.read_csv(train_data_file)
    test_df = pd.read_csv(test_data_file)

    # split features and target for train and test data

    X_train = train_df.drop(columns=["grade"])
    y_train = train_df["grade"]

    X_test = test_df.drop(columns=["grade"])
    y_test = test_df["grade"]

    # feature transformation
    # camis: dropped, these are unique identifiers
    # dba: dropped
    # boro: OHE on categorical variable 
    # zipcode: OHE (taking 20 most frequent zipcodes, everything else is made 'Others')
    # cuisine_description: OHE (taking 35 most frequent cuisine_description, everything else into 'Others'; based on threshold of 600 records)
    # inspection_date: dropped, not relevant
    # action: OHE, only 5 categories
    # violation_code: OHE (top 30, everything else into 'Others'; based on most frequent)
    # violation_description: text with CountVectorizer(); maybe n-gram
    # critical_flag: OHE
    # score: passthrough feature
    # inspection_type: dropped, not relevant - may introduce noise

    categorical_features = ['boro', 'zipcode', 'cuisine_description', 'action', 'violation_code', 'violation_description', 'critical_flag']
    passthrough_features = ['score']
    drop_features = ['camis', 'dba', 'inspection_date', 'inspection_type']
    text_features = 'violation_description'

    # column transformer
    preprocessor = make_column_transformer( 
        ("passthrough", passthrough_features),  
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),  
        (CountVectorizer(stop_words="english"), text_features),
        ("drop", drop_features)
    )
    
    # cross validations for dummy, logistic regression and svm classifier

    print("Performing cross validations for dummy, logistic regression and svm classifier...")
    cross_val_results = {}
    dc = DummyClassifier()
    cross_val_results['dummy'] = pd.DataFrame(cross_validate(dc, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))).agg(['mean', 'std']).round(3).T
    pipe_lr = make_pipeline(preprocessor, LogisticRegression(random_state=123, max_iter=1000))
    cross_val_results['logreg'] = pd.DataFrame(cross_validate(pipe_lr, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))).agg(['mean', 'std']).round(3).T
    pip_svc = make_pipeline(preprocessor, SVC(random_state=123))
    cross_val_results['svc'] = pd.DataFrame(cross_validate(pip_svc, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))).agg(['mean', 'std']).round(3).T

    print(cross_val_results)

    # fitting the logistic regression model to train data because validation scores for LR is higher
    print("Fitting the logistic regression model to train data because validation scores for LR is higher")
    pipe_lr.fit(X_train, y_train)

    # get total length of vocabulary in count vectorizer for 'violation_description' column
    len_vocab_1 = len(pipe_lr.named_steps["columntransformer"].named_transformers_["countvectorizer"].get_feature_names_out())

    # hyper parameter tuning for logistic regression model using randomizedsearchcv
    print("\n Performing hyper parameter tuning for logistic regression model using randomizedsearchcv...")
    param_dist = {'logisticregression__C': loguniform(1e-3, 1e3),
    'columntransformer__countvectorizer__max_features': randint(1, len_vocab_1),
    'logisticregression__class_weight':['balanced', None]}
    random_search = RandomizedSearchCV(pipe_lr, param_dist, n_iter=50, n_jobs=-1, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))
    random_search.fit(X_train, y_train)

    # obtaining the best parameters
    best_parameters = random_search.best_params_

    print("Best parameters found to be: ", best_parameters)

    # transform data using parameters found from randomized cross validation
    preprocessor = make_column_transformer( 
        ("passthrough", passthrough_features),  
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),  
        (CountVectorizer(max_features=best_parameters["columntransformer__countvectorizer__max_features"], stop_words="english"), text_features),
        ("drop", drop_features)
    )
    pipe_lr_best = make_pipeline(preprocessor, LogisticRegression(C=best_parameters["logisticregression__C"], class_weight=best_parameters["logisticregression__class_weight"], random_state=123, max_iter=1000))

    # cross validation on the best logistic regression model

    print("\nDoing cross validation using the best parameters...")
    cross_val_results['logreg_best'] = pd.DataFrame(cross_validate(pipe_lr_best, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))).agg(['mean', 'std']).round(3).T

    print(pd.concat(cross_val_results, axis=1))

    # fit the best model on the training data

    print("\nFitting the best model on training data...")
    pipe_lr_best.fit(X_train, y_train)

    # score the best model on the test data
    score = pipe_lr_best.score(X_test, y_test)
    print("Score on test data : ", score)
    
    # saving the model
    filename = 'finalized_model.sav'
    pickle.dump(pipe_lr_best, open(output_dir_path + filename, 'wb'))

    # to load the model
    # loaded_model = pickle.load(open(output_dir_path + filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)

if __name__ == "__main__":
    main(opt["--train_data"], opt["--test_data"], opt["--output_test_file"])