"""This script downloads the data from the web(given URL) and writes it to the local file system.

Usage: download_csv.py --input_url=<input_url> --output_file=<output_file>

Options:
--input_url=<input_url>        The url which is hosting the data that we are trying to download. This must be in CSV format.
--output_file=<output_file>    Path of the output file which will contain the CSV data once downloaded.
"""
 
import requests
import os.path
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(input_url, output_file):

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

  try:  
    data = pd.read_csv(input_url)
  except Exception as ex:
    print("Exception occurred :" + ex)
    exit()
  
  try:
    isFileExist = os.path.isdir(os.path.dirname(output_file))
    if not isFileExist:
      print("File path does not exist! Creating the path!")
      os.makedirs(os.path.dirname(output_file))
    
    data.to_csv(output_file, index = False)
  except Exception as ex:
    print("Exception occurred :" + ex)
    exit()

if __name__ == "__main__":
  main(opt["--input_url"], opt["--output_file"])
  

