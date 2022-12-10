# author: Nikita Susan Easow and Lauren Zung
# date: 2022-11-24

"""
Takes the preprocessed training and test data and does cross validation with logistic regression and svm classifiers.
Finds logistic regression to be the better model and does hyperparameter tuning to get the best hyperparameters.
Fits the best model on the unseen data (test dataset).

Due to the size of the dataset, this analysis may take up to 5 minutes to run.
   
Usage: src/nyc_rest_analysis.py --train_data=<train_input_file> --test_data=<test_input_file> --output_dir=<output_directory>

Options:
--train_data=<train_input_file>       Path of the input file that contains the train data
--test_data=<test_input_file>         Path of the input file that contains the test data
--output_dir=<output_directory>       Path of the output file where results of the analysis will be stored 

Command to run the script:
python src/nyc_rest_analysis.py --train_data='./data/processed/train_df.csv' --test_data='./data/processed/test_df.csv' --output_dir='./results'
"""
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
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.utils.fixes import loguniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import dataframe_image as dfi
import pickle
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

opt = docopt(__doc__)

def main(train_data, test_data, output_dir):
    """
    Takes the preprocessed training and test data and performs model training
    (using cross validation) with logistic regression and svm classifier
    
    Finds logistic regression to be the better model and does hyperparameter tuning
    to get the best hyperparameter values. It then fits this trained model on the
    unseen data (test_data)

    Parameters
    ----------
    train_data : string
        Relative path of the file that contains the preprocessed training data
    test_data : string
        Relative path of the file which contains the preprocessed testing data
    output_dir : string
        Relative path of the directory where results will be shared
    
    Returns
    --------
    Results table of mean train/validation scores from all models
    Results table of standard deviation in train/validation scores from all models
    Hyperparameter tuning results on optimal classifier
    Train/validation scores from the best model
    Classification report from the best model on the test set
    Confusion matrices from the best model on train and test set
    PR curve from test set
    ROC curve from test set
    Top 20 positive/negative coefficient plot from the best model
    Best model saved as a pickle file
    """
    # suppress all warning messages
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Verify that results directory exists; if not, creates a new folder
    try:
        isDirExist = os.path.isdir(output_dir)
        if not isDirExist:
            print("Directory does not exist. Creating a new folder...")
            os.makedirs(output_dir)
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()

    # Verify that train and test data have been loaded
    isTrainExist = os.path.exists(train_data)
    if not isTrainExist:
        print('Training data has not been added.')
        exit()

    isTestExist = os.path.exists(test_data)
    if not isTestExist:
        print('Testing data has not been added')
        exit()


    # read train and test data from csv files
    print("\nReading data from CSV files...", train_data, test_data)
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    # downsample the training set
    # as we have a large dataset, we have chosen to apply downsampling since we do not have access to enough resources to run the analysis
    print("\nResampling the data, then splitting into X and y...")
    train_df = resample(train_df, replace=False, n_samples=30000, random_state=123)
    test_df = resample(test_df, replace=False, n_samples=10000, random_state=123)

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
    zipcode: Since there are so many restaurants with the same zipcodes, we would OHE it
    cuisine_description: We will OHE the cuisine types as they are categorical in nature
    inspection_date: We would assume that the date of the inspection is unrelated to how restaurants are graded, so we drop the 'inspection_date' feature.
    action: We will drop this column since it is not relevant to the target.
    violation_code: We would use OHE on since there are a fixed number of codes (with appropriate values for max_categories to select the most frequent 20)
    violation_description: We would use Bag of Words for the text with CountVectorizer()
    critical_flag: We would use OHE since a flag can only be Critical, Non-Critical or Not Applicable
    score: Since the score is the only numeric feature, we do not have to apply scaling (score is kept since it is not necessarily correlated with grade)
    inspection_type: We would drop the 'inspection_type' feature since we expect that it does not relate to the target.
    """

    categorical_features = ['boro', 'zipcode', 'cuisine_description', 'violation_code', 'violation_description', 'critical_flag']
    passthrough_features = ['score']
    drop_features = ['camis', 'dba', 'inspection_date', 'action', 'inspection_type']
    text_features = 'violation_description'

    # column transformer
    preprocessor = make_column_transformer( 
        ("passthrough", passthrough_features),  
        (OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=100), categorical_features),  # limit categories to gauge baseline performance
        (CountVectorizer(stop_words="english"), text_features),
        ("drop", drop_features)
    )
    
    # cross validations for dummy, logistic regression and svm classifier
    classification_metrics = {"accuracy" : 'accuracy',
                              "precision" : make_scorer(precision_score, pos_label='F'),
                              "recall" : make_scorer(recall_score, pos_label='F'),
                              "f1" : make_scorer(f1_score, pos_label='F')}

    print("\nPerforming cross validations for dummy, logistic regression and svm classifier (balanced and unbalanced (this may take up to 5 minutes)...")
    cross_val_results = {}
    dc = DummyClassifier()
    print('Dummy')
    cross_val_results['dummy'] = pd.DataFrame(cross_validate(dc, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    print('LogisticRegression')
    pipe_lr = make_pipeline(preprocessor, LogisticRegression(random_state=123, max_iter=1000))
    cross_val_results['logreg'] = pd.DataFrame(cross_validate(pipe_lr, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    print('SVC')
    pipe_svc = make_pipeline(preprocessor, SVC(random_state=123))
    cross_val_results['svc'] = pd.DataFrame(cross_validate(pipe_svc, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    print('LogisticRegression balanced')
    pipe_bal_lr = make_pipeline(preprocessor, LogisticRegression(class_weight="balanced", random_state=123, max_iter=1000))
    cross_val_results['logreg_bal'] = pd.DataFrame(cross_validate(pipe_bal_lr, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    print('SVC balanced')
    pipe_bal_svc = make_pipeline(preprocessor, SVC(class_weight="balanced", random_state=123))
    cross_val_results['svc_bal'] = pd.DataFrame(cross_validate(pipe_bal_svc, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    
    # Style of the header for the tables
    styles = [dict(selector="caption", props=[("font-size", "120%"),
                                          ("font-weight", "bold")])]

    # Adapted from 573 Lab 1
    avg_results_table = pd.concat(
        cross_val_results,
        axis='columns'
    ).xs(
        'mean',
        axis='columns',
        level=1
    ).drop(['fit_time', 'score_time'])
    avg_results_table = avg_results_table.style.format(precision=2).background_gradient(axis=None).set_caption('Table 2.1. Mean train and validation scores from each model.').set_table_styles(styles)
    dfi.export(avg_results_table, output_dir + "/mean_scores_table.png", table_conversion='matplotlib')

    # Adapted from 573 Lab 1
    std_results_table = pd.concat(
        cross_val_results,
        axis='columns'
    ).xs(
        'std',
        axis='columns',
        level=1
    ).drop(['fit_time', 'score_time'])
    std_results_table = std_results_table.style.format(precision=2).background_gradient(axis=None).set_caption('Table 2.2. Standard deviation of train and validation scores for each model.').set_table_styles(styles)
    dfi.export(std_results_table, output_dir + "/std_scores_table.png", table_conversion='matplotlib')

    # fitting the logistic regression model to train data because mean validation score for LR is higher
    print("\nFitting the balanced logistic regression model...")
    pipe_bal_lr.fit(X_train, y_train)

    # get total length of vocabulary in count vectorizer for 'violation_description' column
    len_vocab = len(pipe_bal_lr.named_steps["columntransformer"].named_transformers_["countvectorizer"].get_feature_names_out())

    print("\nPerforming hyperparameter tuning for logistic regression model using RandomizedSearchCV...")
    param_dist = {'logisticregression__C' : loguniform(1e-3, 1e3),
                  'columntransformer__countvectorizer__max_features' : randint(1, len_vocab),
                  'columntransformer__onehotencoder__max_categories' : randint(10, 50)}
    random_search = RandomizedSearchCV(pipe_bal_lr, param_dist, n_iter=20, n_jobs=-1, random_state=123, return_train_score=True, scoring=make_scorer(f1_score, pos_label='F'))
    print("Fitting the optimized model")
    random_search.fit(X_train, y_train)

    # Creating hyperparameter tuning table
    random_cv_df = pd.DataFrame(random_search.cv_results_)[['mean_train_score', 'mean_test_score', 'param_logisticregression__C',
                                                            'param_columntransformer__countvectorizer__max_features',
                                                            'param_columntransformer__onehotencoder__max_categories', 'rank_test_score']].set_index("rank_test_score").sort_index()
    random_cv_df = random_cv_df.style.set_caption('Table 2.3. Mean train and cross-validation scores (5-fold) for balanced logistic regression, optimizing F1 score.').set_table_styles(styles)
    dfi.export(random_cv_df, output_dir + "/hyperparam_results.png", table_conversion='matplotlib')

    print("\nDoing cross validation using the best parameters...")
    best_model_table = pd.DataFrame(cross_validate(random_search.best_estimator_, X_train, y_train, return_train_score=True, scoring=classification_metrics)).agg(['mean', 'std']).round(3).T
    best_model_table = best_model_table.drop(['fit_time', 'score_time'])
    best_model_table = best_model_table.style.set_caption(
        'Table 2.4. Mean and standard deviation of train and validation scores for the balanced logistic regression model.\nParameters: C = ' +
        str(random_search.best_params_['logisticregression__C']) +
        ', max_features = ' + str(random_search.best_params_['columntransformer__countvectorizer__max_features']) +
        ', max_categories = ' + str(random_search.best_params_['columntransformer__onehotencoder__max_categories'])
    ).set_table_styles(styles)
    dfi.export(best_model_table, output_dir + "/best_model_results.png", table_conversion='matplotlib')

    # Create classification report
    print('Creating classification report for the test set...')
    class_report = pd.DataFrame(classification_report(y_test, random_search.best_estimator_.predict(X_test), output_dict=True, digits=3)).T
    class_report = class_report.style.set_caption('Table 2.5. Classification report on the test set.').set_table_styles(styles)
    dfi.export(class_report, output_dir + "/test_classification_report.png", table_conversion='matplotlib')

    # Create confusion matrices for the train and test sets
    print('Creating confusion matrices...')
    cm_plot = plt.figure()
    cm_ax1 = cm_plot.add_subplot(1,2,1)
    cm_ax2 = cm_plot.add_subplot(1,2,2)
    cm_ax1.title.set_text('Train Set')
    cm_ax2.title.set_text('Test Set')
    ConfusionMatrixDisplay.from_predictions(y_train, random_search.best_estimator_.predict(X_train), ax=cm_ax1, colorbar=False)
    ConfusionMatrixDisplay.from_predictions(y_test, random_search.best_estimator_.predict(X_test), ax=cm_ax2, colorbar=False)
    cm_plot.savefig(output_dir + '/confusion_matrices.png')

    # Create and save PR curve for the best model
    print('Creating PR curve...')
    pr_curve, pr_ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(random_search.best_estimator_, X_test, y_test, pos_label='F', ax=pr_ax)
    pr_ax.plot(
        recall_score(y_test, random_search.best_estimator_.predict(X_test), pos_label="F"),
        precision_score(y_test, random_search.best_estimator_.predict(X_test), pos_label="F"),
        "or",
        markersize=10,
        label="Current Threshold",
    )
    pr_ax.legend(loc="best")
    pr_curve.savefig(output_dir + '/PR_curve.png')

    # Create and save ROC curve for the best model
    print('Creating ROC curve...')
    roc_curve, roc_ax = plt.subplots()
    RocCurveDisplay.from_estimator(random_search.best_estimator_, X_test, y_test, pos_label='F', ax=roc_ax)
    roc_ax.legend(loc="best")
    roc_curve.savefig(output_dir + '/ROC_curve.png')

    # saving the model
    print('Exporting the best model...')
    pickle.dump(random_search.best_estimator_, open(output_dir + '/best_model.pkl', 'wb'))

    # Run tests to make sure everything has been saved properly
    avg_score_table_exists(output_dir)
    std_score_table_exists(output_dir)
    hp_tuning_table_exists(output_dir)
    best_model_cv_table_exists(output_dir)
    classification_report_exists(output_dir)
    confusion_matrix_exists(output_dir)
    PR_curve_exists(output_dir)
    ROC_curve_exists(output_dir)
    model_exists(output_dir)

### TESTS
def avg_score_table_exists(file_path):
    """
    Checks that the mean score table has been saved
    """
    assert os.path.isfile(file_path + "/mean_scores_table.png"), "Could not find the average scores table in the results folder." 

def std_score_table_exists(file_path):
    """
    Checks that the score standard deviation table has been saved
    """
    assert os.path.isfile(file_path + "/std_scores_table.png"), "Could not find the std scores table in the results folder." 

def hp_tuning_table_exists(file_path):
    """
    Checks that the hyperparameter tuning table has been saved
    """
    assert os.path.isfile(file_path + "/hyperparam_results.png"), "Could not find the hyperparameter tuning table in the results folder." 

def best_model_cv_table_exists(file_path):
    """
    Checks that the best model's cross-validation results has been saved
    """
    assert os.path.isfile(file_path + "/best_model_results.png"), "Could not find the best model results table in the results folder." 

def classification_report_exists(file_path):
    """
    Checks that the class table has been saved
    """
    assert os.path.isfile(file_path + "/test_classification_report.png"), "Could not find the classification report in the results folder." 

def confusion_matrix_exists(file_path):
    """
    Checks that the confusion matrices have been saved
    """
    assert os.path.isfile(file_path + "/confusion_matrices.png"), "Could not find the confusion matrices in the results folder." 

def PR_curve_exists(file_path):
    """
    Checks that the PR curve has been saved
    """
    assert os.path.isfile(file_path + "/PR_curve.png"), "Could not find the PR curve in the results folder." 

def ROC_curve_exists(file_path):
    """
    Checks that the ROC curve has been saved
    """
    assert os.path.isfile(file_path + "/ROC_curve.png"), "Could not find the ROC curve in the results folder." 

def model_exists(file_path):
    """
    Checks that the best model has been saved
    """
    assert os.path.isfile(file_path + "/best_model.pkl"), "Could not find the model in the results folder." 

if __name__ == "__main__":
    main(opt["--train_data"], opt["--test_data"], opt["--output_dir"])
