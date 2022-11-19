Exploratory Data Analysis of the NYC Restaurant Inspections Data Set
==========================

### DSCI 522 Project - Group 18

##### Team Members: Nikita Susan Easow, Sneha Sunil, Edward (Yukun) Zhang, Lauren Zung

The code associated with this analysis can be found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/src/nyc_rest_eda.ipynb). This report is a summary of our findings.

# Summary of the Data

The data set, DOHMH New York City Restaurant Inspection Results, used in this project is sourced from NYC OpenData Portal. It was retrieved from the `tidytuesday` repository by Thomas Mock, and can be sourced [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-11). The original data set can be found [here](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data).

It contains the violation citations from every inspection conducted for restaurants in New York City from 2012-2018. Each row represents a restaurant that has been assessed by a health inspector, including information about their business such as the restaurant name, phone number, location (borough, building number, street, zip code) and type of cuisine, as well as the details about their inspection (e.g. date, violation code, description, whether there were any violations cited, whether they were critical, etc.). The restaurants can be assigned an official grade of A, B, or C, otherwise they are assigned Z or P for pending review.

# Data Cleaning & Target

There are 300,000 inspections logged in the data set, but only 151,451 of them have a value assigned to the `grade` column. 149,885 of them have been assigned grades as follows:

|Grade|Number of Inspections|
|----------:|--------------:|
|A (Grade A)|119647|
|B (Grade B)|19215|
|C (Grade C)|5888|
|Z (Grade Pending)|3316|
|P (Grade Pending issued on re-opening following an initial inspection that resulted in a closure)|1819|

**Table 1.** Counts of inspections belonging to each class.

There is a large class imbalance (79.8% of inspections are graded as A, thankfully). Thus, we have decided to conduct our analysis as a binary classification problem, where we are interested in determining whether a restaurant passes or fails according to our standards - Grade A vs Grade B/C (Grade F)! We are more interested in predicting if a restaurant will receive below an A grade, so that we can safely avoid eating there. Therefore, we will have to perform further processing during our analysis to overcome the discrepancy (i.e. increasing the importance of the Grade F class).

Additionally, we have restaurants that are "pending" a grade. We will keep these out of our analysis to use as deployment data, where we will see if we can categorize them as either A or F based on their feature values.

# Splitting the Data

Prior to the rest of the analysis, we will divide the data such that 75% of the inspections will be used to train our model(s) and 25% will be for testing to assess performance on unseen restaurants.

| Data Partition | Grade A | Grade F |
|---------------:|--------:|--------:|
|Train           |89781    |18781    |
|Test            |29866    |6322     |

**Table 2.** Counts of inspections belonging to each class in the training and testing sets.

# Exploring the Training Set

Considering the data attributes, we would expect the score and critical flag assigned to an inspection would be good predictors of whether the restaurant will be graded A or not. Thus, we have plotted their distributions by class to investigate whether our assumptions are true:

![Score Boxplot](nyc_rest_eda_figures/score_boxplot.png)

**Figure 1.** Boxplot of the distribution of inspection scores across grades. Green represents Grade A restaurants and orange represents Grade F (below Grade A) restaurants.

It seems that Grade F restaurants are associated with higher scores on average, though some Grade F inspections also received low scores (nearly 10,000 are < 20). We can interpret the score as being higher for more severe/critical health violations, but there does not seem to be a standard cut-off for when a restaurant is considered Grade A or not.

![Violations Plot](nyc_rest_eda_figures/violation_stack.png)

**Figure 2.** Proportion of restaurants that received critical (red) and non-critical (blue) violations by grade. Violations that are unclassified received a 'Not Applicable' flag (green).

We observe a similar relationship with the assignment of critical flags. Grade F restaurants receive proportionately more critical flags as expected, though almost 50% of Grade A restaurants had critical violations during their inspection! It is not clear what the threshold for a "critical" violation is, thus it will be interesting to see whether our model(s) can identify if the severity of a violation actually matters for grading.

![Borough Plot](nyc_rest_eda_figures/boro_bars.png)

**Figure 3.** Number of inspections performed in each NYC borough by grade, where green represents Grade A and orange represents Grade F. The 'Missing' category contains 5 records which are all Grade A.

Each of the NYC boroughs contain mostly Grade A restaurants, thus we should be able to safely eat in any area. We can see that most of the inspections were conducted in Manhattan, which also has the most Grade F restaurants compared to all the other boroughs.

|Cuisine Description|Count of Records|
|------------------:|---------------:|
|American|                                                            24846|
|Chinese|                                                             10954|
|Cafe/Coffee/Tea|                                                      5697|
|Pizza|                                                                4828|
|Italian|                                                              4497|
|Latin (Cuban, Dominican, Puerto Rican, South & Central American)|     4346|
|Mexican|                                                              4123|
|Japanese|                                                             3707|
|Caribbean|                                                            3340|
|Bakery|                                                               3323|

**Table 3.** Number of inspections performed for the top 10 most common cuisine types.

The majority of restaurants in NYC serve American food, followed by Chinese food. The data set also includes food locations that are not necessarily "restaurants" (cafes, bakeries, etc.). Of the 84 unique cuisine descriptions, there are 25 cuisine types that appear less than 100 times in the training set.

![Code Plot](nyc_rest_eda_figures/violation_code_bars.png)

**Figure 4.** Number of inspections designated under each violation code. Bars are coloured by grade where orange is Grade F and green is Grade A.

Interestingly, there does not seem to be specific codes that are uniquely associated with either Grade A nor F. This is why it will be necessary to parse the violation descriptions for certain language that may be independent from an inspection's code. If we can find what words may be common to each class (vermin, roaches, washing, etc.), we can extract the grading for a restaurant based on its description and other feature values.

## Additional Figures

Supplementary charts can be found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/src/nyc_rest_eda.ipynb) on other attributes that were not discussed in detail above.

# References

Mock, T (2022). *Tidy Tuesday: A weekly data project aimed at the R ecosystem.* https://github.com/rfordatascience/tidytuesday.

NYC Open Data Portal (2022). *DOHMH New York City Restaurant Inspection Results.* https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data