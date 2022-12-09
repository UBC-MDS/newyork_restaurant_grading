# author: Lauren Zung
# date: 2022-11-24

"""Creates new tables and plots of the training data from the DOHMH New York City Restaurant Inspection Results as part of the project's exploratory data analysis.
Saves the tables and plots as png files.

Original data source: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv

Usage: src/nyc_rest_eda.py --train_set=<train_set> --visual_dir=<visual_dir>

Options:
--train_set=<train_set>           Path to the training data (in data/preprocessed)
--visual_dir=<visual_dir>         Path to the output directory for the tables and plots (src/nyc_rest_eda_script_visuals)

Command to run the script:
python src/nyc_rest_eda.py --train_set='./data/processed/train_df.csv' --visual_dir='src/nyc_rest_eda_script_visuals'
"""

# Import libraries and packages
import pandas as pd
import altair as alt
import vl_convert as vlc
import dataframe_image as dfi
import matplotlib
import os
from docopt import docopt
import warnings

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
    train_set : string
        The relative path (.csv) that contains the training data
    visual_dir : string
        The name of the directory that will contain the EDA plots and tables
    
    Returns
    -------
    TABLES
        - Counts of each class in the training set
        - Most common cuisine descriptions in the training set
    FIGURES
        - Boxplot of the distribution of scores by grade
        - Stacked bar chart of the critical flags by grade (normalized)
        - Grouped bar chart of the boroughs in NYC and number of inspections performed in each by grade
        - Stacked bar chart of the violation codes by grade
    
    """
    # Suppress warning messages
    warnings.filterwarnings("ignore")

    # Check if the visualization directory exists; if it doesn't, create new folder.
    try:
        isDirExist = os.path.isdir(visual_dir)
        if not isDirExist:
            print("Directory does not exist. Creating a new folder...")
            os.makedirs(visual_dir)
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()

    # Verify that training data has been loaded
    isTrainExist = os.path.exists(train_set)
    if not isTrainExist:
        print('Training data has not been added.')
        exit()
    
    # read in the training data
    print('Training data has been partitioned.')
    train_df = pd.read_csv(train_set)

    # Style of the header for the tables
    styles = [dict(selector="caption", props=[("font-size", "120%"), ("font-weight", "bold"), ("font-family", 'DejaVu Sans')]),
              dict(selector="table", props=[("font-family" , 'DejaVu Sans')])]
    matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # Creates a table of the counts of Grade A and F in the training set
    class_table = train_df['grade'].value_counts().rename_axis('Grades').to_frame('Number of Inspections')
    class_table = class_table.style.set_caption('Table 1.1 Counts of inspections in the training data by class.').set_table_styles(styles)
    dfi.export(class_table, visual_dir + "/class_table.png", table_conversion='matplotlib')

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
    save_chart(score_boxplot, visual_dir + "/score_boxplot.png", 2)

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
    save_chart(flag_bar_chart, visual_dir + "/critical_flag_stacked.png", 2)

    # Creates a grouped bar chart of the number of inspections performed in each
    # NYC borough, sorted by grade
    boro_bar_chart = (
        alt.Chart(train_df).mark_bar().encode(
        alt.X('grade', axis=alt.Axis(labelAngle=0), title='Grade'),
        alt.Y('count()'),
        alt.Color('grade', scale=alt.Scale(scheme='dark2'), legend=None),
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
    save_chart(boro_bar_chart, visual_dir + "/borough_bars.png", 2)

    # Creates a table of the top 10 cuisine types in the training set, in descending
    # order by frequency
    top_10_cuisine_df = pd.DataFrame(train_df['cuisine_description'].value_counts()[:10])
    top_10_cuisine_df.index.name = 'Cuisine Description'
    top_10_cuisine_df.columns = ['Count of Records']
    top_10_cuisine_df = top_10_cuisine_df.style.set_caption('Table 1.2. Number of inspections performed for the top 10 most common cuisine types.').set_table_styles(styles)
    dfi.export(top_10_cuisine_df, visual_dir + "/top_cuisines.png", table_conversion='matplotlib')

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
    save_chart(vc_bar_plot, visual_dir + "/violation_code_bars.png", 2)

    # Run tests to verify that the visuals saved
    class_table_exists(visual_dir)
    score_boxplot_exists(visual_dir)
    flag_plot_exists(visual_dir)
    borough_bar_plot_exists(visual_dir)
    cuisine_table_exists(visual_dir)
    violation_plot_exists(visual_dir)

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

### TESTS
def class_table_exists(file_path):
    """
    Checks that the class table has been saved
    """
    assert os.path.isfile(file_path + "/class_table.png"), "Could not find the class table in the visualizations folder." 

def score_boxplot_exists(file_path):
    """
    Checks that the score boxplot has been saved
    """
    assert os.path.isfile(file_path + "/score_boxplot.png"), "Could not find the score boxplot in the visualizations folder." 

def flag_plot_exists(file_path):
    """
    Checks that the critical flag chart has been saved
    """
    assert os.path.isfile(file_path + "/critical_flag_stacked.png"), "Could not find the critical flag chart in the visualizations folder." 

def borough_bar_plot_exists(file_path):
    """
    Checks that the borough bar plot has been saved
    """
    assert os.path.isfile(file_path + "/borough_bars.png"), "Could not find the borough bar plot in the visualizations folder." 

def cuisine_table_exists(file_path):
    """
    Checks that the cuisine table has been saved
    """
    assert os.path.isfile(file_path + "/top_cuisines.png"), "Could not find the top 10 cuisine table in the visualizations folder." 

def violation_plot_exists(file_path):
    """
    Checks that the violation code plot has been saved
    """
    assert os.path.isfile(file_path + '/violation_code_bars.png'), "Could not find the violation codes chart in the visualizations folder." 

# Call main
if __name__ == "__main__":
    main(opt["--train_set"], opt["--visual_dir"])