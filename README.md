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
### **TO-DO:** Edit once the ML analysis has been completed.

Since the start of the pandemic, hundreds of new restaurants have opened across New York City after the state announced the return of indoor dining (Eater NY, 2020). As government restrictions lift and the hospitality industry opens its doors once again, more and more people are choosing to dine out. Considering the uncertainty of the current time, the overall safety of restaurants has become of paramount importance. Health regulations have become stricter, and it will likely be necessary for health inspectors to reassess the standards that they apply for grading. Although they can differ by state, the general scheme applied by health agencies is as follows:

>>>
GRADE A: The restaurant is clean, up to code, and free of violations.
<br/>
GRADE B: The restaurant has some issues that must be fixed.
<br/>
GRADE C: The restaurant is a public risk and on verge of closure.
<br/>
(Source: SmartSense, 2018)
>>>
  
We chose the dataset, DOHMH New York City Restaurant Inspection Results sourced from 
NYC OpenData Portal. It is retrieved from the tidytuesday repository by Thomas Mock, 
and can be sourced [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-1.).
The original data set can be found [here](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data).

It contains the violation citations from every inspection conducted for restaurants in New York City from 2012 to 2018. Each row represents a restaurant that has been assessed by a health inspector, including information about their business such as the restaurant name, phone number, location (borough, building number, street, zip code) and type of cuisine, as well as the details about their inspection (e.g. date, violation code, description, whether there were any violations cited, whether they were critical, etc.). The restaurants can be assigned an official grade of A, B, or C, otherwise they are assigned Z or P for pending review. A comprehensive dictionary of the data can be found [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-11#data-dictionary).

## Report

This will be updated once the project is finished.

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

    ```python src/nyc_rest_eda.py --train_set='./data/processed/train_df.csv' --visual_dir="src/nyc_rest_eda_script_visuals"```

6. Run the machine learning analysis and export models

    ```python```

7. Render the final report
    ```Rscript -e "rmarkdown::render()"```

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
