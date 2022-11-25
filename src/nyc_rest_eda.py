# author: Lauren Zung
# date: 2022-11-24

"""Creates new tables and plots of the training data from the DOHMH New York City Restaurant Inspection Results as part of the project's exploratory data analysis.
Saves the tables and plots as png files.

Original data source: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv

Usage: src/nyc_rest_eda.py --train_set=<train_set> --visual_dir=<visual_dir>

Options:
--train_set=<train_set>           Path to the training data (in data/preprocessed)
--visual_dir=<visual_dir>         Path to the output directory for the tables and plots (src/nyc_rest_eda_visuals)

Command to run the script:
python src/nyc_rest_eda.py --train_set='./data/processed/train_df.csv' --visual_dir="src/nyc_rest_eda_script_visuals"
"""

# Import libraries and packages
import pandas as pd
import altair as alt
import vl_convert as vlc
import dataframe_image as dfi
import os
from docopt import docopt

# Render figures given large data set
alt.data_transformers.disable_max_rows()

# Initialize doc
opt = docopt(__doc__)

### HELPER FUNCTIONS
#  Added from Joel's suggestions
def save_chart(chart, filename, scale_factor=1):
    """
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    """
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")

# Define main function
def main(train_set, visual_dir):
    """
    Creates and saves all tables and figures from the EDA

    Parameters
    ----------
    train_set : csv file
        The relative path that contains the training data, as a string
    visual_dir : visualization directory
        The relative path that will contain the EDA plots, as a string

    """
    # Check if the visualization directory exists; if it doesn't, create new folder
    try:
        isDirExist = os.path.isdir(visual_dir)
        if not isDirExist:
            print("Directory does not exist. Creating a new folder...")
            os.makedirs(visual_dir)
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()
    
    # read in the training data
    train_df = pd.read_csv(train_set)

    # Creates a table of the counts of Grade A and F in the training set
    class_table = train_df['grade'].value_counts().rename_axis('Grades').to_frame('Number of Inspections')
    class_table = class_table.style.set_caption('Table 1. Counts of inspections in the training data by class.')
    dfi.export(class_table, visual_dir + "/class_table.png")

    # Creates boxplot of the distribution of scores by grade
    score_boxplot = (
        alt.Chart(train_df).mark_boxplot(size=50).encode(
        alt.X('score', title='Score'),
        alt.Y('grade', title='Grade'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), title='Grade')
        ).properties(
            title="Distribution of Scores by Grade",
            height=300,
            width=500
        ).configure_title(
            anchor='start',
            fontSize=22
        ).configure_axis(
            titleFontSize=18
        )
    )
    save_chart(score_boxplot, visual_dir + "/score_boxplot.png")

    # Creates stacked bar chart of the proportion of restaurants that received
    # Critical, Non-critical and Not Applicable flags
    flag_bar_chart = (
        alt.Chart(train_df).mark_bar().encode(
        alt.X('count()', stack='normalize', title='Proportion of Restaurants'),
        alt.Y('grade', title='Grade'),
        alt.Color('critical_flag', scale=alt.Scale(scheme='set1'), title='Critical Flag', sort='-x')
        ).properties(
            title='Severity of Violations',
            height=200
        ).configure_title(
            anchor='start',
            fontSize=22
        ).configure_axis(
            titleFontSize=18
        )
    )
    save_chart(flag_bar_chart, visual_dir + "/critical_flag_stacked.png")

    # Creates a grouped bar chart of the number of inspections performed in each
    # NYC borough, sorted by grade
    boro_bar_chart = (
        alt.Chart(train_df).mark_bar().encode(
        alt.X('grade', axis=alt.Axis(labelAngle=0), title='Grade'),
        alt.Y('count()'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), title='Grade', legend=None),
        alt.Column('boro', title=None, header=alt.Header(titleFontSize=12), sort=["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND", "Missing"])
        ).properties(
            height=300,
            width=100,
            title='Number of Inspections Conducted by NYC Borough'
        ).configure_title(
            anchor='start',
            fontSize=25
        ).configure_axis(
            titleFontSize=18
        )
    )
    save_chart(boro_bar_chart, visual_dir + "/borough_bars.png")

    # Creates a table of the top 10 cuisine types in the training set, in descending
    # order by frequency
    top_10_cuisine_df = pd.DataFrame(train_df['cuisine_description'].value_counts()[:10])
    top_10_cuisine_df.index.name = 'Cuisine Description'
    top_10_cuisine_df.columns = ['Count of Records']
    top_10_cuisine_df = top_10_cuisine_df.style.set_caption('Table 2. Number of inspections performed for the top 10 most common cuisine types.')
    dfi.export(top_10_cuisine_df, visual_dir + "/top_cuisines.png")

    # Creates a bar plot of the number of inspections categorized under each violation code,
    # sorted by grade
    vc_bar_plot = (
        alt.Chart(train_df).mark_bar().encode(
        alt.X('violation_code', sort='-y', axis=alt.Axis(labelAngle=0), title='Violation Code'),
        alt.Y('count()'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), title='Grade')
        ).properties(
            title='Violation Codes by Grade',
            height=500
        ).configure_title(
            anchor='start',
            fontSize=25
        ).configure_axis(
            labelFontSize=10,
            titleFontSize=18
        )
    )
    save_chart(vc_bar_plot, visual_dir + "/violation_code_bars.png")

    # Run tests to verify that the visuals saved
    class_table_exists(visual_dir + "/class_table.png")
    score_boxplot_exists(visual_dir + "/score_boxplot.png")
    flag_plot_exists(visual_dir + "/critical_flag_stacked.png")
    borough_bar_plot_exists(visual_dir + "/borough_bars.png")
    cuisine_table_exists(visual_dir + "/top_cuisines.png")
    violation_plot_exists(visual_dir + '/violation_code_bars.png')

### TESTS
def class_table_exists(file_path):
    assert os.path.isfile(file_path), "Could not find the class table in the visualizations folder." 

def score_boxplot_exists(file_path):
    assert os.path.isfile(file_path), "Could not find the score boxplot in the visualizations folder." 

def flag_plot_exists(file_path):
    assert os.path.isfile(file_path), "Could not find the critical flag chart in the visualizations folder." 

def borough_bar_plot_exists(file_path):
    assert os.path.isfile(file_path), "Could not find the borough bar plot in the visualizations folder." 

def cuisine_table_exists(file_path):
    assert os.path.isfile(file_path), "Could not find the top 10 cuisine table in the visualizations folder." 

def violation_plot_exists(file_path):
    assert os.path.isfile(file_path), "Could not find the violation codes chart in the visualizations folder." 

# Call main
if __name__ == "__main__":
    main(opt["--train_set"], opt["--visual_dir"])