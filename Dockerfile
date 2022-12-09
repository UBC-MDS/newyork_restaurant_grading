FROM continuumio/miniconda3:4.12.0

# Update list of available software packages & install make
RUN apt update && apt install -y make

# install dev tools
RUN apt-get install -y libxml2-dev libcurl4-openssl-dev libssl-dev
RUN apt install -y libfontconfig1-dev
RUN apt-get install gcc python3-dev chromium-driver -y
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt install ./google-chrome-stable_current_amd64.deb -y

# Override miniconda python installation
RUN conda install -c conda-forge -c defaults \
    'python=3.9.*' \
    'ipykernel' \
    'ipython>=7.15' \
    'vega_datasets=0.9.0' \
    'altair_saver' \
    'selenium<4.3.0' \
    'scikit-learn>=1.0' \
    'pandas>=1.3.*' \
    'requests>=2.24.0' \
    'dataframe_image=0.1.1' \
    'scipy=1.9.3' \
    'matplotlib=3.6.2' \ 
    'matplotlib-base=3.6.2' \
    'matplotlib-inline=0.1.6'

# Fix fonts
# RUN rm -fr ~/.cache/matplotlib
# RUN conda install -c conda-forge -y mscorefonts
# RUN apt install font-manager -y

RUN pip install \
    'docopt-ng==0.8.*' \
    'joblib==1.1.*' \
    'mglearn' \
    'psutil>=5.7.2' \
    'vl-convert-python==0.5.*' \
    'lxml'

# R pre-requisites
RUN apt-get install r-base r-base-dev -y

# Install R packages
RUN Rscript -e \
    "install.packages('tidyverse')" \
    "install.packages('kableExtra')" \
    "install.packages(c('rmarkdown', 'here'), repos = 'https://mran.revolutionanalytics.com/snapshot/2022-12-05')" \
    "install.packages('knitr', version = '1.29.*')"
