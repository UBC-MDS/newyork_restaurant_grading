# NYC Restaurants Data Analysis Pipeline
# Author: Nikita Susan Easow, Sneha Sunil, Edward (Yukun) Zhang, Lauren Zung
# Date: 2022-12-01

# This driver script completes the analysis of grading prediction for restaurants in NYC
# and creates 6 figures and tables for the EDA and 9 plots for the results to generate the final html report. 
# This script takes no arguments.

# Example usage:
# make all
# make clean

all : doc/ny_rest_report.html

# download the data via url
data/raw/nyc_restaurants.csv : src/download_csv.py
	python src/download_csv.py --input_url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv" --output_file="./data/raw/nyc_restaurants.csv"

# preprocess the data (train test split)
data/processed/test_df.csv data/processed/train_df.csv : src/pre_process_nyc_rest.py ./data/raw/nyc_restaurants.csv
	python src/pre_process_nyc_rest.py --input_file="./data/raw/nyc_restaurants.csv" --output_train_file="./data/processed/train_df.csv" --output_test_file="./data/processed/test_df.csv"

# run the exploratory data analysis and generate intermediate tables and plots
src/nyc_rest_eda_script_visuals/borough_bars.png src/nyc_rest_eda_script_visuals/class_table.png src/nyc_rest_eda_script_visuals/critical_flag_stacked.png src/nyc_rest_eda_script_visuals/score_boxplot.png src/nyc_rest_eda_script_visuals/top_cuisines.png src/nyc_rest_eda_script_visuals/violation_code_bars.png : src/nyc_rest_eda.py ./data/processed/train_df.csv 
	python src/nyc_rest_eda.py --train_set='./data/processed/train_df.csv' --visual_dir='src/nyc_rest_eda_script_visuals'

# build the model and generate results tables and plots
results/PR_curve.png results/ROC_curve.png results/best_model.pkl results/best_model_results.png results/confusion_matrices.png results/hyperparam_results.png results/mean_scores_table.png results/std_scores_table.png results/test_classification_report.png results/violation_coefs.png : src/nyc_rest_analysis.py ./data/processed/train_df.csv ./data/processed/test_df.csv
	python src/nyc_rest_analysis.py --train_data='./data/processed/train_df.csv' --test_data='./data/processed/test_df.csv' --output_dir='./results'

# write the report
doc/ny_rest_report.html : doc/ny_rest_report.Rmd src/nyc_rest_eda_script_visuals/borough_bars.png src/nyc_rest_eda_script_visuals/class_table.png src/nyc_rest_eda_script_visuals/critical_flag_stacked.png src/nyc_rest_eda_script_visuals/score_boxplot.png src/nyc_rest_eda_script_visuals/top_cuisines.png src/nyc_rest_eda_script_visuals/violation_code_bars.png results/PR_curve.png results/ROC_curve.png results/best_model.pkl results/best_model_results.png results/confusion_matrices.png results/hyperparam_results.png results/mean_scores_table.png results/std_scores_table.png results/test_classification_report.png results/violation_coefs.png
	Rscript -e "rmarkdown::render('doc/ny_rest_report.Rmd', output_format = 'html_document')"

clean :
    rm -f results/*
    rm -f data/processed/*.csv
    rm -f data/raw/*.csv
    rm -f src/nyc_rest_eda_script_visuals/*
    rm -f doc/*.html
