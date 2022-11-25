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