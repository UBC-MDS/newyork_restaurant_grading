# author: Nikita Susan Easow
# date: 2022-11-24

"""
Takes the preprocessed training and test data and does cross validation with logistic regression and svm classifier - finds logistic regression to be the better model and does hyperparameter tuning to get the best hyperparameters. It then fits this trained model on the unseen data (test dataset).
   
Usage: src/fit_new_york_grading_prediction_model.py --train_data=<train_input_file> --test_data=<test_input_file> --output_dir=<output_directory>
Options:
--train_data=<train_input_file>       Path of the input file that contains the train data
--test_data=<test_input_file>         Path of the input file that contains the test data
--output_dir=<output_directory>       Path of the output file where results of the analysis will be stored 
Command to run the script:
python src/fit_new_york_grading_prediction_model.py --train_data="./data/processed/train_df.csv" --test_data="./data/processed/test_df.csv" --output_dir="./results/"
"""

# REFERENCE : code to plot PR curve referred from 573 lecture 1 notes

from docopt import docopt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, recall_score, precision_score
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
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os

opt = docopt(__doc__)

def main(train_data, test_data, output_dir):
    output_dir = output_dir[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    """
    Takes the preprocessed training and test data and does cross validation with logistic regression and svm classifier - finds logistic regression to be the better model and does hyperparameter tuning to get the best hyperparameters. It then fits this trained model on the unseen data (test dataset).
    
    Parameters
    ----------
    string : train_data
        Relative path of the file that contains the preprocessed training data
    string : test_data
        Relative path of the file which contains the preprocessed testing data
    string : output_dir
        Relative path of the directory where results will be share
    
    """ 
    # read train and test data from csv files

    print("Reading data from CSV files...", train_data, test_data)
    train_df = pd.read_csv(train_data[0])
    test_df = pd.read_csv(test_data[0])

    # split features and target for train and test data

    X_train = train_df.drop(columns=["grade"])
    y_train = train_df["grade"]

    X_test = test_df.drop(columns=["grade"])
    y_test = test_df["grade"]

    # Feature Transformations :
    
    # camis: We drop this feature because these are unique identifiers
    # dba: We would drop the 'dba' since we expect the words in name feature of the restaurants to be unrealted to the grading.
    # boro: We will use OHE on the restaurant regions which is a categorical variable 
    # zipcode: Since there are so many restaurants with the same zipcodes, we would OHE it (with appropriate values for max_categories to select the most frequent 20)
    # cuisine_description: OHE on the descriptions (there are not many words) which is a categorical variable
    # inspection_date: We would assume that the date of the inspection is unrealted to how restaurants are graded, so we drop the 'inspection_date' feature.
    # action: We would use OHE on categorical variable
    # violation_code: We would use OHE on categorical variable
    # violation_description: We would use Bag of Words for the text with CountVectorizer()
    # critical_flag: We would use OHE on categorical variable
    # score: Since 'score' is the ONLY numeric feature, We would not apply any transformation on it (no need to do scaling).
    # inspection_type: We would drop the 'inspection_type' feature since we expect it does not relate to the grading target.

    categorical_features = ['boro', 'zipcode', 'cuisine_description', 'action', 'violation_code', 'violation_description', 'critical_flag']
    passthrough_features = ['score']
    drop_features = ['camis', 'dba', 'inspection_date', 'inspection_type']
    text_features = 'violation_description'

    # column transformer
    preprocessor = make_column_transformer( 
        ("passthrough", passthrough_features),  
        (OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20), categorical_features),  
        (CountVectorizer(stop_words="english", max_features=2000), text_features),
        ("drop", drop_features)
    )
    
    # cross validations for dummy, logistic regression and svm classifier

    print("Performing cross validations for dummy, logistic regression and svm classifier...")
    cross_val_results = {}
    dc = DummyClassifier()
    cross_val_results['dummy'] = cross_validate(dc, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))
    pipe_lr = make_pipeline(preprocessor, LogisticRegression(random_state=123, max_iter=2000))
    cross_val_results['logreg'] = cross_validate(pipe_lr, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))
    pip_svc = make_pipeline(preprocessor, SVC(random_state=123))
    cross_val_results['svc'] = cross_validate(pip_svc, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))

    print(cross_val_results)

    # fitting the logistic regression model to train data because validation scores for LR is higher
    print("Fitting the logistic regression model to train data because validation scores for LR is higher")
    pipe_lr.fit(X_train, y_train)

    # get total length of vocabulary in count vectorizer for 'violation_description' column
    len_vocab_1 = len(pipe_lr.named_steps["columntransformer"].named_transformers_["countvectorizer"].get_feature_names_out())

    print("\nPerforming hyper parameter tuning for logistic regression model using randomizedsearchcv...")
    param_dist = {'logisticregression__C': loguniform(1e-3, 1e3),
    'columntransformer__countvectorizer__max_features': randint(1, len_vocab_1),
    'logisticregression__class_weight':['balanced', None],
    "logisticregression__solver" : ["sag"]}
    random_search = RandomizedSearchCV(pipe_lr, param_dist, n_iter=10, n_jobs=-1, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))
    print("Fitting the optimized model")
    random_search.fit(X_train, y_train)

    # obtaining the best parameters
    best_parameters = random_search.best_params_

    print("Best parameters found to be: ", best_parameters)

    # transform data using parameters found from randomized cross validation
    preprocessor = make_column_transformer( 
        #("passthrough", passthrough_features),  
        (OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20), categorical_features),  
        (CountVectorizer(max_features=best_parameters["columntransformer__countvectorizer__max_features"], stop_words="english"), text_features),
        ("drop", drop_features)
    )
    pipe_lr_best = make_pipeline(preprocessor, LogisticRegression(C=best_parameters["logisticregression__C"], class_weight=best_parameters["logisticregression__class_weight"], random_state=123, max_iter=2500, solver='sag'))

    # cross validation on the best logistic regression model

    print("\nDoing cross validation using the best parameters...")
    cross_val_results['logreg_best'] = cross_validate(pipe_lr_best, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))

    for i, j in cross_val_results.items():
        for l,m in j.items():
            j[l] = m.mean()
    final_cross_val_results = pd.DataFrame(cross_val_results)
    print(final_cross_val_results)

    # fit the best model on the training data

    print("\nFitting the best model on training data...")
    pipe_lr_best.fit(X_train, y_train)

    # score the best model on the test data
    score = pipe_lr_best.score(X_test, y_test)
    print("Score on test data : ", score)

    # create and save PR curve for the best model

    print("\nCreating and saving PR curve plot...")
    precision, recall, thresholds = precision_recall_curve(
    y_test, pipe_lr.predict_proba(X_test)[:, 1], pos_label="F"
    )
    plt.plot(precision, recall, label="logistic regression: PR curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot(
        precision_score(y_test, pipe_lr.predict(X_test), pos_label="F"),
        recall_score(y_test, pipe_lr.predict(X_test), pos_label="F"),
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best");
    plt.savefig(output_dir + 'logistic_regression_PR_curve.png')
    
    # saving the model
    filename = 'finalized_model.sav'
    pickle.dump(pipe_lr, open(output_dir + filename, 'wb'))

    # to load the model
    # loaded_model = pickle.load(open(output_dir + filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)

if __name__ == "__main__":
    main(opt["--train_data"], opt["--test_data"], opt["--output_dir"])