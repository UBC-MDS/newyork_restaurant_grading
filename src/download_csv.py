"""This script downloads the data from the web(given URL) and writes it to the local file system.
   This has been developed based on the script available at - https://github.com/ttimbers/breast_cancer_predictor/blob/master/src/download_data.py

Usage: download_csv.py --input_url=<input_url> --output_file=<output_file>

Options:
--input_url=<input_url>        The url which is hosting the data that we are trying to download. This must be in CSV format.
--output_file=<output_file>    Path of the output file which will contain the CSV data once downloaded(File name must be included).

Command to run the script:
python src/download_csv.py --input_url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-12-11/nyc_restaurants.csv" --output_file="./data/raws/nyc_restaurants.csv"

"""
 
import requests
import os.path
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(input_url, output_file):

  """This function takes an input URL and 
    downloads the data locally

    Parameters
    ----------
    string : input_url
        The input URL which contains the data
    string : output_file
        The output file which will have the downloaded data   
    """ 

  # Check if the website is existing or not
  try: 
    response = requests.get(input_url)
    if response.status_code == 200:
      print("Provided URL is valid. Proceeding to read the data!")
    else:
      print("Provided URL is invalid. Please provide correct URL!")
      exit()
  except Exception as ex:
    print("Exception occurred :" + ex)
    exit()

  # Read the data from the URL 
  try:  
    data = pd.read_csv(input_url)
  except Exception as ex:
    print("Exception occurred :" + ex)
    exit()
  
  # Check if the directory exists. If it doesn't create new folder and download the data
  try:
    isFileExist = os.path.isdir(os.path.dirname(output_file))
    if not isFileExist:
      print("Directory does not exist! Creating the path!")
      os.makedirs(os.path.dirname(output_file))
    
    data.to_csv(output_file, index = False)
  except Exception as ex:
    print("Exception occurred :" + ex)
    exit()

if __name__ == "__main__":
  main(opt["--input_url"], opt["--output_file"])
  

