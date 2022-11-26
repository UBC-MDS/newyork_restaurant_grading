# author: Nikita Susan Easow and Lauren Zung
# date: 2022-11-24

"""
Takes the preprocessed training and test data and does cross validation with logistic regression and svm classifier - finds logistic regression to be the better model and does hyperparameter tuning to get the best hyperparameters. It then fits this trained model on the unseen data (test dataset).
   
Usage: src/nyc_rest_results.py --train_data=<train_input_file> --test_data=<test_input_file> --output_dir=<output_directory>

Options:
--train_data=<train_input_file>       Path of the input file that contains the train data
--test_data=<test_input_file>       Path of the input file that contains the test data
--output_dir=<output_directory>       Path of the output file where results of the analysis will be stored 
Command to run the script:
python src/nyc_rest_results.py --train_data='./data/processed/train_df.csv' --test_data='./data/processed/test_df.csv' --output_dir='./results'
"""

# REFERENCE : code to plot PR curve referenced from 573 lecture 1 notes

from docopt import docopt
import pandas as pd
from sklearn.utils import resample
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from scipy.stats import randint
import dataframe_image as dfi
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os

opt = docopt(__doc__)

def main(train_data, test_data, output_dir):
    """
    Takes the preprocessed training and test data and performs model training
    (using cross validation) with logistic regression and svm classifier
    
    Finds logistic regression to be the better model and does hyperparameter tuning
    to get the best hyperparameter values. It then fits this trained model on the
    unseen data (test_data)

    Outputs:
        - the model saved as a pickle file
        - results table of mean train/test scores from all models
        - PR curve
        - AOC curve
        - confusion matrix from the best model
    
    Parameters
    ----------
    train_data : string
        Relative path of the file that contains the preprocessed training data
    test_data : string
        Relative path of the file which contains the preprocessed testing data
    output_dir : string
        Relative path of the directory where results will be shared
    
    """ 
    # Verify that results directory exists; if not, creates a new folder
    try:
        isDirExist = os.path.isdir(output_dir)
        if not isDirExist:
            print("Directory does not exist. Creating a new folder...")
            os.makedirs(output_dir)
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()

    # read train and test data from csv files

    print("Reading data from CSV files...", train_data, test_data)
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    # downsample the training set
    # as we have a large dataset, we have chosen to apply downsampling since we do not have access to enough resources to run the analysis
    print("Resampling the data, then splitting into X and y...")
    train_df = resample(train_df, replace=False, n_samples=20000, random_state=123)

    # split features and target for train and test data

    X_train = train_df.drop(columns=["grade"])
    y_train = train_df["grade"]

    X_test = test_df.drop(columns=["grade"])
    y_test = test_df["grade"]

    """
    # Feature Transformations :
    camis: We drop this feature because these are unique identifiers
    dba: We would drop the 'dba' since we expect the words in name feature of the restaurants to be unrelated to the grading.
    boro: We will use OHE on the restaurant regions which is categorical in nature 
    zipcode: Since there are so many restaurants with the same zipcodes, we would OHE it (with appropriate values for max_categories to select the most frequent 20)
    cuisine_description: We will OHE the cuisine types as they are categorical in nature
    inspection_date: We would assume that the date of the inspection is unrelated to how restaurants are graded, so we drop the 'inspection_date' feature.
    action: We will use OHE since there are a fixed number of actions.
    violation_code: We would use OHE on since there are a fixed number of codes (with appropriate values for max_categories to select the most frequent 20)
    violation_description: We would use Bag of Words for the text with CountVectorizer()
    critical_flag: We would use OHE since a flag can only be Critical, Non-Critical or Not Applicable
    score: Since the score is the only numeric feature, we do not have to apply scaling (score is kept since it is not necessarily correlated with grade)
    inspection_type: We would drop the 'inspection_type' feature since we expect that it does not relate to the target.
    """

    categorical_features = ['boro', 'zipcode', 'cuisine_description', 'action', 'violation_code', 'violation_description', 'critical_flag']
    passthrough_features = ['score']
    drop_features = ['camis', 'dba', 'inspection_date', 'inspection_type']
    text_features = 'violation_description'

    # column transformer
    preprocessor = make_column_transformer( 
        ("passthrough", passthrough_features),  
        (OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20), categorical_features),  
        (CountVectorizer(stop_words="english"), text_features),
        ("drop", drop_features)
    )
    
    # cross validations for dummy, logistic regression and svm classifier
    classification_metrics = {"accuracy" : 'accuracy',
                              "precision" : make_scorer(precision_score, pos_label='F'),
                              "recall" : make_scorer(recall_score, pos_label='F'),
                              "f1" : make_scorer(f1_score, pos_label='F')}

    print("Performing cross validations for dummy, logistic regression and svm classifier (balanced and unbalanced)...")
    cross_val_results = {}
    dc = DummyClassifier()
    cross_val_results['dummy'] = pd.DataFrame(cross_validate(dc, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    pipe_lr = make_pipeline(preprocessor, LogisticRegression(random_state=123, max_iter=1000))
    cross_val_results['logreg'] = pd.DataFrame(cross_validate(pipe_lr, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    pipe_svc = make_pipeline(preprocessor, SVC(random_state=123))
    cross_val_results['svc'] = pd.DataFrame(cross_validate(pipe_svc, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    pipe_bal_lr = make_pipeline(preprocessor, LogisticRegression(class_weight="balanced", random_state=123, max_iter=1000))
    cross_val_results['logreg_bal'] = pd.DataFrame(cross_validate(pipe_bal_lr, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    pipe_bal_svc = make_pipeline(preprocessor, SVC(class_weight="balanced", random_state=123))
    cross_val_results['svc_bal'] = pd.DataFrame(cross_validate(pipe_bal_svc, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T

    # Adapted from 573 Lab 1
    avg_results_table = pd.concat(
        cross_val_results,
        axis='columns'
    ).xs(
        'mean',
        axis='columns',
        level=1
    ).drop(['fit_time', 'score_time'])
    avg_results_table = avg_results_table.style.format(precision=2).background_gradient(axis=None).set_caption('Table 1. Mean train and test scores of each model.')
    dfi.export(avg_results_table, output_dir + "/mean_scores_table.png")

#     # fitting the logistic regression model to train data because validation scores for LR is higher
#     print("Fitting the logistic regression model to train data because validation scores for LR is higher")
#     pipe_lr.fit(X_train, y_train)

#     # get total length of vocabulary in count vectorizer for 'violation_description' column
#     len_vocab_1 = len(pipe_lr.named_steps["columntransformer"].named_transformers_["countvectorizer"].get_feature_names_out())

#     print("\nPerforming hyper parameter tuning for logistic regression model using randomizedsearchcv...")
#     param_dist = {'logisticregression__C': loguniform(1e-3, 1e3),
#     'columntransformer__countvectorizer__max_features': randint(1, len_vocab_1),
#     'logisticregression__class_weight':['balanced', None],
#     "logisticregression__solver" : ["sag"]}
#     random_search = RandomizedSearchCV(pipe_lr, param_dist, n_iter=10, n_jobs=-1, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))
#     print("Fitting the optimized model")
#     random_search.fit(X_train, y_train)

#     # obtaining the best parameters
#     best_parameters = random_search.best_params_

#     print("Best parameters found to be: ", best_parameters)

#     # transform data using parameters found from randomized cross validation
#     preprocessor = make_column_transformer( 
#         #("passthrough", passthrough_features),  
#         (OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20), categorical_features),  
#         (CountVectorizer(max_features=best_parameters["columntransformer__countvectorizer__max_features"], stop_words="english"), text_features),
#         ("drop", drop_features)
#     )
#     pipe_lr_best = make_pipeline(preprocessor, LogisticRegression(C=best_parameters["logisticregression__C"], class_weight=best_parameters["logisticregression__class_weight"], random_state=123, max_iter=2500, solver='sag'))

#     # cross validation on the best logistic regression model

#     print("\nDoing cross validation using the best parameters...")
#     cross_val_results['logreg_best'] = cross_validate(pipe_lr_best, X_train, y_train, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))

#     for i, j in cross_val_results.items():
#         for l,m in j.items():
#             j[l] = m.mean()
#     final_cross_val_results = pd.DataFrame(cross_val_results)
#     print(final_cross_val_results)

#     # fit the best model on the training data

#     print("\nFitting the best model on training data...")
#     pipe_lr_best.fit(X_train, y_train)

#     # score the best model on the test data
#     score = pipe_lr_best.score(X_test, y_test)
#     print("Score on test data : ", score)

#     # create and save PR curve for the best model

#     print("\nCreating and saving PR curve plot...")
#     precision, recall, thresholds = precision_recall_curve(
#     y_test, pipe_lr.predict_proba(X_test)[:, 1], pos_label="F"
#     )
#     plt.plot(precision, recall, label="logistic regression: PR curve")
#     plt.xlabel("Precision")
#     plt.ylabel("Recall")
#     plt.plot(
#         precision_score(y_test, pipe_lr.predict(X_test), pos_label="F"),
#         recall_score(y_test, pipe_lr.predict(X_test), pos_label="F"),
#         "or",
#         markersize=10,
#         label="threshold 0.5",
#     )
#     plt.legend(loc="best");
#     plt.savefig(output_dir + 'logistic_regression_PR_curve.png')
    
#     # saving the model
#     filename = 'finalized_model.sav'
#     pickle.dump(pipe_lr, open(output_dir + filename, 'wb'))

#     # to load the model
#     # loaded_model = pickle.load(open(output_dir + filename, 'rb'))
#     # result = loaded_model.score(X_test, Y_test)

if __name__ == "__main__":
    main(opt["--train_data"], opt["--test_data"], opt["--output_dir"])