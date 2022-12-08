FROM continuumio/miniconda3:4.12.0

# Update list of available software packages & install make
RUN apt update && apt install -y make

# Override miniconda python installation
RUN conda install -c conda-forge -c defaults \
    'python=3.11.0' \
    'ipykernel' \
    'ipython=8.6.0' \
    'vega_datasets=0.9.0' \
    'altair_saver' \
    'selenium=4.2.0' \
    'scikit-learn=1.1.3' \
    'pandas=1.5.2' \
    'requests=2.28.1' \
    'dataframe_image=0.1.1' \
    'scipy=1.9.3' \
    'matplotlib=3.6.2' \
    'matplotlib-base=3.6.2' \
    'matplotlib-inline=0.1.6'

RUN pip install \
    'docopt-ng==0.8.*' \
    'joblib==1.1.*' \
    'mglearn==0.1.9' \
    'psutil>=5.7.2' \
    'vl-convert-python==0.5.*'

# R pre-requisites
RUN apt-get install r-base r-base-dev -y

# Install R packages
RUN conda install -c conda-forge --quiet --yes \
    'r-base=4.0.3' \
    'r-tidyverse=1.3*' \
    'r-rmarkdown=2.5*' \
    'r-knitr=1.29.*'