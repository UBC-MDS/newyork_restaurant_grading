
FROM continuumio/miniconda3

RUN apt-get update && apt install -y make

RUN apt-get install r-base r-base-dev -y

RUN Rscript -e "install.packages('knitr')"



