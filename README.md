# Predicting the Grade of Restaurants in New York City

- Authors (ordered alphabetically by last name) from Group 18:
    - Nikita Susan Easow
    - Sneha Sunil
    - Edward (Yukun) Zhang
    - Lauren Zung

A data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

URL of the project repo: https://github.com/UBC-MDS/newyork_restaurant_grading.git

## Project Summary

In this project, we build a classification model using logistic regression and support vector machines which uses health inspection data to predict whether a restaurant will be graded A (i.e., the restaurant is clean, up to code, and free of violations.) or F (i.e., the restaurant has some issues that must be fixed or is a public risk on the verge of closure).

Our best model was a balanced logistic regressor with a C value of 0.024947, 1 numeric feature, 130 text features and 47 categorical features. On a test set of 10000 samples, we returned an F1 score of 0.999 and precision and recall scores of 0.999 and 0.999 respectively, indicating that our model is highly effective at classifying both grade A and F restaurants. We also computed the area under a receiver operating characteristic curve which was found to be 1.00. This is the optimum value which also supports that the predictions from our model are close to 100% correct.

We chose the dataset, DOHMH New York City Restaurant Inspection Results sourced from NYC OpenData Portal. It is retrieved from the tidytuesday repository by Thomas Mock, and can be sourced here. The original data set can be found here. It contains the violation citations from every inspection conducted for restaurants in New York City from 2012 to 2018. Each row represents a restaurant that has been assessed by a health inspector, including information about their business such as the restaurant name, phone number, location and type of cuisine, as well as the details about their inspection. The restaurants can be assigned an official grade of A, B, or C, otherwise they are assigned Z or P for pending review.

## Report

[Here](https://ubc-mds.github.io/newyork_restaurant_grading/doc/ny_rest_report.html) is the link to the Project Report.

## Usage

To replicate this analysis, clone this GitHub repository and download the environment found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/environment.yaml) to install the necessary [dependencies](#dependencies).

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

    ```conda activate nyc_rest```

Run the following command at the command line/terminal from the root directory of this project:

    make all
    
To reset the repo to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:

    make clean

## Dependencies

Note: more packages are likely to be added in future updates/milestones.

The associated environment with all dependencies required for this project can be found [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/environment.yaml).

Python 3.11.0 and the following packages:
  - aiohttp=3.8.3
  - aiosignal=1.3.1
  - altair=4.2.0
  - altair_data_server=0.4.1
  - altair_saver=0.1.0
  - altair_viewer=0.4.0
  - appnope=0.1.3
  - asttokens=2.1.0
  - async-timeout=4.0.2
  - async_generator=1.10
  - attrs=22.1.0
  - backcall=0.2.0
  - backports=1.0
  - backports.functools_lru_cache=1.6.4
  - beautifulsoup4=4.11.1
  - bleach=5.0.1
  - brotli=1.0.9
  - brotli-bin=1.0.9
  - brotlipy=0.7.0
  - bzip2=1.0.8
  - ca-certificates=2022.9.24
  - certifi=2022.9.24
  - cffi=1.15.1
  - charset-normalizer=2.1.1
  - comm=0.1.1
  - contourpy=1.0.6
  - cryptography=38.0.3
  - cycler=0.11.0
  - dataframe_image=0.1.1
  - debugpy=1.6.3
  - decorator=5.1.1
  - defusedxml=0.7.1
  - docopt=0.6.2
  - entrypoints=0.4
  - executing=1.2.0
  - fonttools=4.38.0
  - freetype=2.12.1
  - frozenlist=1.3.3
  - h11=0.14.0
  - idna=3.4
  - importlib-metadata=5.1.0
  - importlib_resources=5.10.0
  - ipython=8.6.0
  - jedi=0.18.2
  - jinja2=3.1.2
  - jpeg=9e
  - jsonschema=4.17.1
  - jupyter_client=7.4.7
  - jupyter_core=5.0.0
  - jupyterlab_pygments=0.2.2
  - kiwisolver=1.4.4
  - lcms2=2.14
  - lerc=4.0.0
  - libblas=3.9.0
  - libbrotlicommon=1.0.9
  - libbrotlidec=1.0.9
  - libbrotlienc=1.0.9
  - libcblas=3.9.0
  - libcxx=14.0.6
  - libdeflate=1.14
  - libffi=3.4.2
  - libgfortran=5.0.0
  - libgfortran5=11.3.0
  - liblapack=3.9.0
  - libopenblas=0.3.21
  - libpng=1.6.39
  - libsodium=1.0.18
  - libsqlite=3.40.0
  - libtiff=4.4.0
  - libwebp-base=1.2.4
  - libxcb=1.13
  - libzlib=1.2.13
  - llvm-openmp=15.0.5
  - markupsafe=2.1.1
  - matplotlib=3.6.2
  - matplotlib-base=3.6.2
  - matplotlib-inline=0.1.6
  - mistune=2.0.4
  - multidict=6.0.2
  - munkres=1.1.4
  - nbclient=0.7.0
  - nbconvert=7.2.5
  - nbconvert-core=7.2.5
  - nbconvert-pandoc=7.2.5
  - nbformat=5.7.0
  - ncurses=6.3
  - nest-asyncio=1.5.6
  - numpy=1.23.5
  - openjpeg=2.5.0
  - openssl=3.0.7
  - outcome=1.2.0
  - packaging=21.3
  - pandas=1.5.2
  - pandoc=2.19.2
  - pandocfilters=1.5.0
  - parso=0.8.3
  - pexpect=4.8.0
  - pickleshare=0.7.5
  - pillow=9.2.0
  - pip=22.3.1
  - pkgutil-resolve-name=1.3.10
  - platformdirs=2.5.2
  - portpicker=1.5.2
  - prompt-toolkit=3.0.33
  - psutil=5.9.4
  - pthread-stubs=0.4
  - ptyprocess=0.7.0
  - pure_eval=0.2.2
  - pycparser=2.21
  - pygments=2.13.0
  - pyopenssl=22.1.0
  - pyparsing=3.0.9
  - pyrsistent=0.19.2
  - pysocks=1.7.1
  - python-dateutil=2.8.2
  - python-fastjsonschema=2.16.2
  - python_abi=3.11
  - pytz=2022.6
  - pyzmq=24.0.1
  - readline=8.1.2
  - requests=2.28.1
  - scikit-learn=1.1.3
  - scipy=1.9.3
  - selenium=4.2.0
  - setuptools=65.5.1
  - six=1.16.0
  - sniffio=1.3.0
  - sortedcontainers=2.4.0
  - soupsieve=2.3.2.post1
  - stack_data=0.6.1
  - threadpoolctl=3.1.0
  - tinycss2=1.2.1
  - tk=8.6.12
  - toolz=0.12.0
  - tornado=6.2
  - traitlets=5.5.0
  - trio=0.22.0
  - trio-websocket=0.9.2
  - typing-extensions=4.4.0
  - typing_extensions=4.4.0
  - tzdata=2022f
  - urllib3=1.26.13
  - vega_datasets=0.9.0
  - wcwidth=0.2.5
  - webencodings=0.5.1
  - wheel=0.38.4
  - wsproto=1.2.0
  - xorg-libxau=1.0.9
  - xorg-libxdmcp=1.1.3
  - xz=5.2.6
  - yarl=1.8.1
  - zeromq=4.3.4
  - zipp=3.10.0
  - zstd=1.5.2

## License

The New York City Restaurant dataset was adapted from tidytuesday dataset
made available under the license **Creative Commons Zero v1.0 Universal** 
which was originally taken from the Department of Health and Mental Hygiene (DOHMH)
owned by NYC OpenData.

For more details about the License of this project, please click [here](https://github.com/UBC-MDS/newyork_restaurant_grading/blob/main/LICENSE).

## References

<div id="refs" class="references hanging-indent">

<div id="ref-Mock2022">

Mock, T (2022). Tidy Tuesday: A weekly data project aimed at the R ecosystem. https://github.com/rfordatascience/tidytuesday.

</div>

<div id="ref-NYCOpen">

NYC Open Data Portal (2022). DOHMH New York City Restaurant Inspection Results.
https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/data

</div>
    
<div id="ref-SmartSense">
    
SmartSense. (2018, August 17). Restaurant letter grading: What does a B really mean? Connected Insights Blog. Retrieved November 19, 2022, from https://blog.smartsense.co/restaurant-letter-grading#:~:text=GRADE%20A%3A%20The%20restaurant%20is,and%20on%20verge%20of%20closure. 

</div>
    
<div id="ref-EaterNY">
    
Staff, E. (2020, June 10). A running list of new restaurants that opened during the pandemic. Eater NY. Retrieved November 19, 2022, from https://ny.eater.com/2020/6/10/21270665/nyc-new-restaurant-openings-coronavirus 

</div>
    
</div>
