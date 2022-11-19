# Prediction for the Grading of New York City Restaurant

  - authors (ordered alphabetically by last name) from Group 18:
    Nikita Susan Easow, Sneha Sunil, Edward (Yukun) Zhang, Lauren Zung
  

A data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

URL of the project repo: https://github.com/UBC-MDS/newyork_restaurant_grading.git

## Project Proposal

With the development of the metropolitan areas and large cities, more and more people tend to eat outside more frequently and the overall safety of restaurants becomes very important. After all, we are what eat, and as data scientists, we are interested in finding a quantitative way to predict the overall quality of a certain restaurant. If a restaurant can be accurately predicted as "bad", then we can safely avoid eating there. Given the New York City Restaurant Inspection Results dataset, we would like to focus our study on predicting the restaurant grading in New York City based on our target of grading standards. Then if possible, we plan to further explore and see whether our model can generalize for other large cities across the world or not and this data analysis project would have great values for local residents and tourists. 

Besides this main research question, we would also plan to address some interesting sub_questions 
such as the following: 
  - Which cuisines is more likely to be graded A in NYC?
  - Which cuisine is more likely to be graded F in NYC?
  - Which borough in NYC seems to have the best restaurants?
  - Which borough in NYC seem to have the most restaurants with the most code violations?
  
We choose the large dataset DOHMH New York City Restaurant Inspection Results sourced from 
NYC OpenData Portal. It is retrieved from the tidytuesday repository by Thomas Mock, 
and can be sourced [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-1.).
The original data set can be found [here](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data).
It contains the violation citations from every inspection conducted for restaurants in New York City from 2012 to 2018. 
Each row represents a restaurant that has been assessed by a health inspector, 
including information about their business such as the restaurant name, phone number, location (borough, building number, street, zip code) 
and type of cuisine, as well as the details about their inspection 
(e.g. date, violation code, description, whether there were any violations cited, whether they were critical, etc.). 
The restaurants can be assigned an official grade of A, B, or C, otherwise they are assigned Z or P for pending review.

To address our main predictive question above, we plan to build a predictive classification model. At first, we split our data into a training and test set (train-test ratio 75%:25%) and perform exploratory data analysis (EDA) to assess whether there is a strong class imbalance issue for our target grading. The target class counts will be presented as a table and used to identify the existence of the class imbalance problem.
If we have a large class imbalance, we might try to perform our analysis as a binary classification problem by 
combining Grade B/C as Grade F to reduce the class imbalance issue. 

Furthermore, we would like to graphically explore the relations between features and the target in order to choose our features properly. Considering the data attributes, we would expect the score and critical flag assigned to an inspection to be good predictors of whether the restaurant will be graded A or not. 
Thus, we plan to plot their distributions in box plots and bar plots by grading class to investigate whether our assumptions are true or not. If these selected features do not contribute very much to predict our target grading, then we might consider to drop these features and re-modify our models. 

For more details about all the EDA figures and tables in this project, please click [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/src/nyc_rest_eda.md).

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

<div id="ref-Dua2019">

Mock, T (2022). Tidy Tuesday: A weekly data project aimed at the R ecosystem. https://github.com/rfordatascience/tidytuesday.

</div>

<div id="ref-Streetetal">

NYC Open Data Portal (2022). DOHMH New York City Restaurant Inspection Results.
https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data

</div>

</div>
