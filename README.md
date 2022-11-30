# Predicting the Grade of Restaurants in New York City

- Authors (ordered alphabetically by last name) from Group 18:
    - Nikita Susan Easow
    - Sneha Sunil
    - Edward (Yukun) Zhang
    - Lauren Zung

A data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

URL of the project repo: https://github.com/UBC-MDS/newyork_restaurant_grading.git

## Project Summary

In this project, we build a classification model using logistic regression and support vector machines which uses health inspection data to predict whether a restaurant will be graded A (i.e., the restaurant is clean, up to code, and free of violations.) or F (i.e., the restaurant has some issues that must be fixed or is a public risk on the verge of closure).

Our best model was a balanced logistic regressor with a C value of 0.024947, 1 numeric feature, 130 text features and 47 categorical features. On a test set of 10000 samples, we returned an F1 score of 0.999 and precision and recall scores of 0.999 and 0.999 respectively, indicating that our model is highly effective at classifying both grade A and F restaurants. We also computed the area under a receiver operating characteristic curve which was found to be 1.00. This is the optimum value which also supports that the predictions from our model are close to 100% correct.

We chose the dataset, DOHMH New York City Restaurant Inspection Results sourced from NYC OpenData Portal. It is retrieved from the tidytuesday repository by Thomas Mock, and can be sourced here. The original data set can be found here. It contains the violation citations from every inspection conducted for restaurants in New York City from 2012 to 2018. Each row represents a restaurant that has been assessed by a health inspector, including information about their business such as the restaurant name, phone number, location and type of cuisine, as well as the details about their inspection. The restaurants can be assigned an official grade of A, B, or C, otherwise they are assigned Z or P for pending review.

## Report

[Here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/doc/ny_rest_report.pdf) is the link to the Project Report. 

## Usage

To replicate this analysis, clone this GitHub repository and download the environment found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/environment.yaml) to install the necessary [dependencies](#dependencies).

1. Clone the repository
    ```
    git clone git@github.com:UBC-MDS/newyork_restaurant_grading.git
    ```
    
    or
    
    ```
    git clone https://github.com/UBC-MDS/newyork_restaurant_grading.git
    ```

2. Navigate to the repository

    ```
    cd newyork_restaurant_grading
    ```

3. Create the environment

    ```conda env create -f environment.yaml```

    Assuming that the environment was created successfully, you can activate the environment as follows:

    ```conda activate nyc_rest```

Run the following commands at the command line/terminal from the root directory of this project:

4. Download the data

    ```python src/download_csv.py --input_url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv" --output_file="./data/raw/nyc_restaurants.csv"```

4. Process the data

    ```python src/pre_process_nyc_rest.py --input_file="./data/raw/nyc_restaurants.csv" --output_train_file="./data/processed/train_df.csv" --output_test_file="./data/processed/test_df.csv"```

5. Create exploratory data analysis figures and tables

    ```python src/nyc_rest_eda.py --train_set='./data/processed/train_df.csv' --visual_dir='src/nyc_rest_eda_script_visuals'```

6. Run the machine learning analysis and export models

    ```python src/nyc_rest_analysis.py --train_data='./data/processed/train_df.csv' --test_data='./data/processed/test_df.csv' --output_dir='./results'```

7. Render the final report

    ```Rscript -e "rmarkdown::render('doc/ny_rest_report.Rmd', output_format = 'pdf_document')"```
    ```Rscript -e "rmarkdown::render('doc/ny_rest_report.Rmd', output_format = 'html_document')"```

## Dependencies

Note: more packages are likely to be added in future updates/milestones.

The associated environment with all dependencies required for this project can be found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/environment.yaml).


## License

The New York City Restaurant dataset was adapted from tidytuesday dataset
made available under the license **Creative Commons Zero v1.0 Universal** 
which was originally taken from the Department of Health and Mental Hygiene (DOHMH)
owned by NYC OpenData.

For more details about the License of this project, please click [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/LICENSE).

## References

<div id="refs" class="references hanging-indent">

<div id="ref-Mock2022">

Mock, T (2022). Tidy Tuesday: A weekly data project aimed at the R ecosystem. https://github.com/rfordatascience/tidytuesday.

</div>

<div id="ref-NYCOpen">

NYC Open Data Portal (2022). DOHMH New York City Restaurant Inspection Results.
https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data

</div>
    
<div id="ref-SmartSense">
    
SmartSense. (2018, August 17). Restaurant letter grading: What does a B really mean? Connected Insights Blog. Retrieved November 19, 2022, from https://blog.smartsense.co/restaurant-letter-grading#:~:text=GRADE%20A%3A%20The%20restaurant%20is,and%20on%20verge%20of%20closure. 

</div>
    
<div id="ref-EaterNY">
    
Staff, E. (2020, June 10). A running list of new restaurants that opened during the pandemic. Eater NY. Retrieved November 19, 2022, from https://ny.eater.com/2020/6/10/21270665/nyc-new-restaurant-openings-coronavirus 

</div>
    
</div>
