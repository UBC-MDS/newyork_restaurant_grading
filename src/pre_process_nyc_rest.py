# author: Edward Yukun ZHang
# date: 2022-11-24

"""Cleans, splits and pre-processes the New York City Restaurant Grading from "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv"
   Writes the training and test data to separate csv files.
   
Usage: pre_process_nyc_rest.py --input_file=<input_file> --output_train_file=<output_train_file> --output_test_file=<output_test_file>

Options:
--input_file=<input_file>                     Path of the input file that contains the raw data
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
    """This function takes an input file and generate train and test data frame
    Parameters
    ----------
    string : input_file
        Path of the input file that contains the raw data
    string : output_train_file
        Path of the output file which will contain the CSV train data 
    string : output_test_file
        Path of the output file which will contain the CSV test data 
    
    """ 
        
    # Read the data from input
    nyc_df = pd.read_csv(input_file)


    # Clean and drop the NA values (without affecting the imbalance probelm too much)
    nyc_drop_na_df = nyc_df.dropna()
    
    nyc_mod_target_df = nyc_drop_na_df.query("grade == ['A', 'B', 'C']")
    
    # Change the label name of target grading
    nyc_mod_target_df.loc[nyc_mod_target_df['grade'] != 'A', 'grade'] = 'F'
    
    # Create final data
    nyc_final_df = nyc_mod_target_df

    # Train & test split
    train_df, test_df = train_test_split(nyc_final_df, test_size=0.25, random_state=123)
    
    # Transform outputs into csv file
    train_df.to_csv(output_train_file, index = False)
    test_df.to_csv(output_test_file, index = False)
    

if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_train_file"], opt["--output_test_file"])
    
    
