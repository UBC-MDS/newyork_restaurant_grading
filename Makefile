# NYC Restaurants Data Analysis Pipeline
# Author: Nikita Susan Easow, Sneha Sunil, Edward (Yukun) Zhang, Lauren Zung
# Date: 2022-12-01

all : doc/ny_rest_report.html

clean :
    rm -f results/*
    rm -f data/processed/*.csv
    rm -f data/raw/*.csv
    rm -f src/nyc_rest_eda_script_visuals/*
    rm -f doc/*.html