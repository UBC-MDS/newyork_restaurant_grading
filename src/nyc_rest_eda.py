# author: Lauren Zung
# date: 2022-11-24

"""Creates new tables and plots of the preprocessed training data from the DOHMH New York City Restaurant Inspection Results as part of the project's exploratory data analysis.
Saves the tables and plots as png files.

Data source:

Usage: src/nyc_rest_eda.py --train_set=<train_set> --visual_dir=<visual_dir>

Options:
--train_set=<train_set>           Path to the training data (in data/preprocessed)
--visual_dir=<visual_dir>         Path to the output directory for the tables and plots (src/nyc_rest_eda_visuals)
"""

# Import libraries and packages
import pandas as pd
import altair as alt
import vl_convert as vlc
import dataframe_image as dfi
from docopt import docopt

# Render figures given large data set
alt.data_transformers.disable_max_rows()

# Initialize doc
opt = docopt(__doc__)

nyc_df = pd.read_csv("../data/raw/nyc_restaurants.csv")
nyc_df_drop_na = nyc_df[nyc_df['grade'].notna()]
nyc_df_final = nyc_df_drop_na.loc[(nyc_df_drop_na['grade'] == 'A') | (nyc_df_drop_na['grade'] == 'B') | (nyc_df_drop_na['grade'] == 'C')]
nyc_df_final.loc[nyc_df_final['grade'] != 'A', 'grade'] = 'F'

# Define main function
def main(train_set, visual_dir):
    # read in the training data
    train_df = pd.read_csv(train_set)

    data_part_table = data_partition_table(train_df)
    dfi.export(data_part_table, visual_dir + '/data_partition_table.png')

# Call main
if __name__ == "__main__":
    main(opt["--train_set"], opt["--visual_dir"])

### TESTS


### HELPER FUNCTIONS
def data_partition_table(train):
    """
    Create table of the counts of Grade A and F in the training set
    """
    table = pd.DataFrame(train['grade'].value_counts(), columns=['Number of Inspections'])
    table.style.set_caption('Table 1. Counts of inspections belonging to each class in the training data')
    return table

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