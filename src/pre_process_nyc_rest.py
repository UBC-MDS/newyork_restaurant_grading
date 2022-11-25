"""Cleans, splits and pre-processes the New York City Restaurant Grading from "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv"
   Writes the training and test data to separate csv files.
   
Usage: pre_process_nyc_rest.py --input_file=<input_file> --output_train_file=<output_train_file> --output_test_file=<output_test_file>

Options:
--input_file=<input_file>                     Path of the input file from doenload
--output_train_file=<output_train_file>       Path of the output file which will contain the CSV train data 
--output_test_file=<output_test_file>         Path of the output file which will contain the CSV test data 


Command to run the script:
python src/pre_process_nyc_rest.py --input_file="./data/raw/nyc_restaurants.csv" --output_train_file="./data/processed/train_df.csv" --output_test_file="./data/processed/test_df.csv"

"""

import requests
import os.path
from docopt import docopt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

opt = docopt(__doc__)

def main(input_file, output_train_file, output_test_file):

    nyc_df = pd.read_csv(input_file)
    nyc_df.info()


    # Clean and modify the data
    nyc_drop_na_df = nyc_df.dropna()
    nyc_drop_na_df.info()
    nyc_drop_na_df.head()
    nyc_drop_na_df.describe(include='all')
    
    nyc_mod_target_df = nyc_drop_na_df.query("grade == ['A', 'B', 'C']")
    nyc_mod_target_df.loc[nyc_mod_target_df['grade'] != 'A', 'grade'] = 'F'
    nyc_mod_target_df['grade'].value_counts()
    
    nyc_mod_zipcode_df = nyc_mod_target_df.copy()
    nyc_mod_zipcode_df['zipcode'] = nyc_mod_target_df['zipcode'].apply(int).apply(str)
    top_20_zipcode = ['10019', '10003', '10036', '10013', '10001', '10002', '10016', '10022', '10011', '11201', 
                  '11354', '10012', '11220', '10014', '11372', '10017', '10018', '11215', '11211', '10009' ]
    nyc_mod_zipcode_df.loc[nyc_mod_zipcode_df.query("zipcode != @top_20_zipcode").index, 'zipcode'] = 'other_zipcode'
    
    nyc_mod_cuisine_df = nyc_mod_zipcode_df.copy()
    nyc_mod_cuisine_df.loc[nyc_mod_cuisine_df[nyc_mod_cuisine_df['cuisine_description'].map(nyc_mod_cuisine_df['cuisine_description'].value_counts()) < 600].index, 'cuisine_description'] = 'Other_cuisine'
    
    nyc_mod_violation_des_df = nyc_mod_cuisine_df.copy()
    nyc_mod_violation_des_df.loc[nyc_mod_violation_des_df[nyc_mod_violation_des_df['violation_description'].map(nyc_mod_violation_des_df['violation_description'].value_counts()) < 500].index, 'violation_description'] = 'Other_violation_des'
    
    nyc_mod_violation_code_df = nyc_mod_violation_des_df.copy()
    nyc_mod_violation_code_df.loc[nyc_mod_violation_code_df[nyc_mod_violation_code_df['violation_code'].map(nyc_mod_violation_code_df['violation_code'].value_counts()) < 1000].index, 'violation_code'] = 'Other_violation_code'
    
    nyc_final_df = nyc_mod_violation_code_df
    nyc_final_df.head()

    # Train & test split
    train_df, test_df = train_test_split(nyc_final_df, test_size=0.25, random_state=123)
    
    train_df.to_csv(output_train_file, index = False)
    test_df.to_csv(output_test_file, index = False)

if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_train_file"], opt["--output_test_file"])
    
    
