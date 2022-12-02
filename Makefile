# NYC Restaurants Data Analysis Pipeline
# Author: Nikita Susan Easow, Sneha Sunil, Edward (Yukun) Zhang, Lauren Zung
# Date: 2022-12-01

all : doc/ny_rest_report.html

data/raw/nyc_restaurants.csv : src/download_csv.py
	python src/download_csv.py --input_url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv" --output_file="./data/raw/nyc_restaurants.csv"


clean :
    rm -f results/*
    rm -f data/processed/*.csv
    rm -f data/raw/*.csv
    rm -f src/nyc_rest_eda_script_visuals/*
    rm -f doc/*.html