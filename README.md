# Prediction for the Grading of New York City Restaurant

  - authors (ordered alphabetically by last name) from Group 18:
    Nikita Susan Easow, Sneha Sunil, Edward (Yukun) Zhang, Lauren Zung
  

A data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

URL of the project repo: https://github.com/UBC-MDS/newyork_restaurant_grading.git

## Project Proposal

Since the start of the pandemic, hundreds of new restaurants have opened across New York City after the state announced the return of indoor dining (Eater NY, 2020). As government restrictions lift and the hospitality industry opens its doors once again, more and more people are choosing to dine out. Considering the uncertainty of the current time, the overall safety of restaurants has become of paramount importance. Health regulations have become stricter, and it will likely be necessary for health inspectors to reassess the standards that they apply for grading. Although they can differ by state, the general scheme applied by health agencies is as follows:

> GRADE A: The restaurant is clean, up to code, and free of violations.
> GRADE B: The restaurant has some issues that must be fixed.
> GRADE C: The restaurant is a public risk and on verge of closure.
> Source: SmartSense, 2018

As data scientists, we are interested in whether we can accurately assess the overall quality of a restaurant. If a restaurant can be predicted as "good" or "bad" (in our case, Grade A vs Grade B/C), then we can make appropriate recommendations to others. As we currently have access to data on restaurants in New York City, we would like to focus our analysis on predicting the grading for NYC locations, with aims to expand to other metropolitan areas in the future. We feel that this project could bring value to local residents or tourists who are troubled with deciding on where they'd like to eat.

Besides this main research question, our analysis may also address some interesting sub-questions such as the following: 
  - Which cuisines is more likely to be graded A in NYC?
  - Which cuisine is more likely to be graded B or C) in NYC?
  - Which borough in NYC seems to have the best restaurants?
  - Which borough in NYC seems to have the most restaurants with the most severe violations?
  
We choose the dataset, DOHMH New York City Restaurant Inspection Results sourced from 
NYC OpenData Portal. It is retrieved from the tidytuesday repository by Thomas Mock, 
and can be sourced [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-1.).
The original data set can be found [here](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data).

It contains the violation citations from every inspection conducted for restaurants in New York City from 2012 to 2018. Each row represents a restaurant that has been assessed by a health inspector, including information about their business such as the restaurant name, phone number, location (borough, building number, street, zip code) and type of cuisine, as well as the details about their inspection (e.g. date, violation code, description, whether there were any violations cited, whether they were critical, etc.). The restaurants can be assigned an official grade of A, B, or C, otherwise they are assigned Z or P for pending review. A comprehensive dictionary of the data can be found [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-11#data-dictionary).

To address our main predictive question above, we plan to construct a predictive classification model. First, we will split our data into training and testing sets (train-test ratio 75:25). We will then perform exploratory data analysis (EDA) on the training set to assess whether there is a need for our model, as well as address the possible concerns with our target, `grade`. The target class counts will be presented as a table and used to identify the existence of the class imbalance problem. If we have a large class imbalance, we might try to perform our analysis as a binary classification problem by 
combining Grade B/C as Grade F to reduce the class imbalance issue. 

Furthermore, we would like to graphically explore the relations between features and the target in order to choose our features properly. Considering the data attributes, we would expect the score and critical flag assigned to an inspection to be good predictors of whether the restaurant will be graded A or not. 
Thus, we plan to plot their distributions in box plots and bar plots by grading class to investigate whether our assumptions are true or not. If these selected features do not contribute very much to predict our target grading, then we might consider to drop these features and re-modify our models. 

For more details about all the EDA figures and tables in this project, please click [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/src/nyc_rest_eda.ipynb).

After the EDA, we plan to fit several supervised machine learning classification models (KNN, Logistic Regression, SVM and so on), and optimize the corresponding hyperparameters in cross validation to generate the best fitted models. Then we will collect and compare the results across multiple error measurement metrics and visualize the modeling results as a table to generate the report. To better share and improve the quality of our analysis, we would also incorporate the overall accuracy, confusion matrix and PR curve analysis in our report and summarize the final robust version of the report in a single PDF/md file.

## Report

This will be updated once the project is finished. 

## Usage

There is one suggested way to run this analysis:

To replicate the analysis, clone this GitHub repository, install the
[dependencies](##Dependencies) listed below, and run the following
command at the command line/terminal from the root directory of this
project:
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

4. Download the data

    ```python src/download_csv.py --input_url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv" --output_file="./data/raw/nyc_restaurants.csv"```

 

## Dependencies

Note: more packages are likely to be added in future updates/milestones.

  - Channels:
      - conda-forge
      - defaults
  - Dependencies:
      - docopt=0.6.2
      - ipykernel
      - ipython>=7.15
      - vega_datasets
      - altair_saver
      - selenium<4.3.0
      - matplotlib>=3.2.2
      - scikit-learn>=1.0
      - pandas>=1.3.*
      - requests>=2.24.0
      - pip:
        - joblib==1.1.0
        - mglearn
        - psutil>=5.7.2


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
