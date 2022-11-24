# author: Lauren Zung
# date: 2022-11-24

"""Creates new tables and plots of the training data from the DOHMH New York City Restaurant Inspection Results as part of the project's exploratory data analysis.
Saves the tables and plots as png files.

Data source:

Usage: src/nyc_rest_eda.py --train_set=<train_set> --visual_dir=<visual_dir>

Options:
--train_set=<train_set>           Path to the training data (in data/preprocessed)
--visual_dir=<visual_dir>         Path to the output directory for the tables and plots (src/nyc_rest_eda_visuals)

Command to run the script:
python src/nyc_rest_eda.py --train_set='../data/processed/train_set.csv' --visual_dir="./nyc_rest_eda_visuals"
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

# Define main function
def main(train_set, visual_dir):
    """
    Creates and saves all tables and figures from the EDA

    Parameters
    ----------
    train_set : csv file
        The relative path that contains the training data, as a string
    visual_dir : visualization directory
        The relative path that will contain the EDA plots

    """
    # read in the training data
    train_df = pd.read_csv(train_set)
    # Check if the directory exists. If it doesn't create new folder and download the data
    try:
        isDirExist = os.path.isdir(os.path.dirname(visual_dir))
        if not isDirExist:
            print("Directory does not exist! Creating the path!")
            os.makedirs(os.path.dirname(visual_dir))
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()

    class_table(train_df, visual_dir + "/data_partition_table.png")
    score_boxplot(train_df, visual_dir + "/score_boxplot.png")
    flags_stacked_bar_chart(train_df, visual_dir + "/critical_flag_stacked.png")
    borough_bars(train_df, visual_dir + "/borough_bars.png")
    top_cuisine_table(train_df, visual_dir + "/top_cuisines.png")

    # Run tests to verify that the visuals saved
    class_table_exists(visual_dir + "/data_partition_table.png")
    score_boxplot_exists(visual_dir + "/score_boxplot.png")
    flag_plot_exists(visual_dir + "/critical_flag_stacked.png")
    borough_bar_plot_exists(visual_dir + "/borough_bars.png")

# Call main
if __name__ == "__main__":
    main(opt["--train_set"], opt["--visual_dir"])

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

### HELPER FUNCTIONS
def class_table(train, path):
    """
    Creates a table of the counts of Grade A and F in the training set
    """
    table = pd.DataFrame(train['grade'].value_counts(), columns=['Number of Inspections'])
    table.style.set_caption('Table 1. Counts of inspections belonging to each class in the training data')
    return dfi.export(table, path)

def score_boxplot(train, path):
    """
    Creates boxplot of the distribution of scores by grade
    """
    boxplot = (
        alt.Chart(train).mark_boxplot(size=50).encode(
        alt.X('score', title='Score'),
        alt.Y('grade', title='Grade'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), title='Grade')
        ).properties(
        title="Distribution of Scores by Grade",
        height=300,
        width=500
        ).configure_title(anchor='start')
    )
    return save_chart(boxplot, path)

def flags_stacked_bar_chart(train, path):
    """
    Creates stacked bar chart of the proportion of restaurants that received
    Critical, Non-critical and Not Applicable flags
    """
    bar_chart = (
        alt.Chart(train).mark_bar().encode(
        alt.X('count()', stack='normalize', title='Proportion of Restaurants'),
        alt.Y('grade', title='Grade'),
        alt.Color('critical_flag', scale=alt.Scale(scheme='set1'), title='Critical Flag', sort='-x')
        ).properties(title='Severity of Violations', height=200)
    )
    return save_chart(bar_chart, path)

def borough_bars(train, path):
    """
    Creates a grouped bar chart of the number of inspections performed in each
    NYC borough, sorted by grade
    """
    bar_chart = (
        alt.Chart(train).mark_bar().encode(
        alt.X('grade', axis=alt.Axis(labelAngle=0), title='Grade'),
        alt.Y('count()'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), title='Grade', legend=None),
        alt.Column('boro', title=None, sort=["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND", "Missing"])
        ).properties(height=300, width=100, title='Number of Inspections Conducted by NYC Borough')
    )
    return save_chart(bar_chart, path)

def top_cuisine_table(train, path):
    """
    Creates a table of the top 10 cuisine types in the training set, in descending
    order by frequency
    """
    top_10_cuisine_df = pd.DataFrame(train['cuisine_description'].value_counts()[:10])
    top_10_cuisine_df.index.name = 'Cuisine Description'
    top_10_cuisine_df.columns = ['Count of Records']
    return dfi.export(top_10_cuisine_df, path)

def violation_code_bars(train, path):
    bar_plot = (
        alt.Chart(
        train_df,
        ).mark_bar().encode(
        alt.X('violation_code', sort='-y', axis=alt.Axis(labelAngle=0), title='Violation Code'),
        alt.Y('count()'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), title='Grade')
        ).properties(
            title='Violation Codes by Grade', height=500
        ).configure_title(anchor='start')
    )
    return save_chart(bar_plot, path)

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