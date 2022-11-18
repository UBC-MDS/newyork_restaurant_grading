Exploratory Data Analysis of the NYC Restaurant Inspections Data Set
==========================

# Summary of the Data

The data set, DOHMH New York City Restaurant Inspection Results, used in this project is sourced from NYC OpenData Portal. It was sourced from the `tidytuesday` repository by Thomas Mock, and can be sourced [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-12-1.). The original data set can be found [here](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data).

It contains the violation citations from every inspection conducted for restaurants in New York City from 2012-2018. Each row represents a restaurant that has been assessed by a health inspector, including information about their business such as the restaurant name, phone number, location (borough, building number, street, zip code) and type of cuisine, as well as the details about their inspection (e.g. date, violation code, description, whether there were any violations cited, whether they were critical, etc.). The restaurants can be assigned an official grade of A, B, or C.

# Data Cleaning & Target

There are 300,000 restaurants in the data set, but only 151,451 of them have a value assigned to the `grade` column. 149,885 of them have been assigned grades as follows:

|A (Grade A)|B (Grade B)|C (Grade C)|Z (Grade Pending)|P (Grade Pending issued on re-opening following an initial inspection that resulted in a closure)|
|----------:|----------:|----------:|----------------:|----------------:|
|119647|19215|5888|3316|1819|
Table 1. Counts of restaurants belonging to each class.

There is a large class imbalance (79.8% of restaurants are graded as A, thankfully), thus we have decided to conduct our analysis as a binary classification problem where we are interested in determining whether a restaurant passes or fails according to our standards - grade A vs everything else (grade F)!

# Splitting the data into train and test sets

| Data Partition | Grade A | Grade F |
|---------------:|--------:|--------:|
|Train           |89789    |22624    |
|Test            |29858    |7614     |

# References

Mock, T (2022). *Tidy Tuesday: A weekly data project aimed at the R ecosystem.* https://github.com/rfordatascience/tidytuesday.

NYC Open Data Portal (2022). *DOHMH New York City Restaurant Inspection Results.* https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data