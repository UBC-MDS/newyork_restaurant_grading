# newyork_restaurant_grading

## Usage

To replicate this analysis, clone this GitHub repository and download the environment found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/src/environment.yaml) for all necessary dependencies.


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

    ```
    conda activate nyc_rest
    ```

## Dependencies

Note: more packages are likely to be added in future updates/milestones.
- Python 3.9.13 and Python packages:
  - ipykernel
  - matplotlib>=3.2.2
  - scikit-learn>=1.0
  - pandas>=1.3.*
  - requests>=2.24.0
  - graphviz
  - python-graphviz
  - python==3.9.*
  - ipykernel
  - ipython>=7.15
  - vega_datasets
  - altair_saver
  - selenium<4.3.0