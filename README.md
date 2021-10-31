# CoSMo

Implementation of:  
Modeling Dynamic Interaction over Tensor Streams

## Installation

This code is tested in: **Python 3.8.5**  
You will install the following packages with pip or conda

- numpy
- pandas
- tensorly
- termcolor
- tqdm
- lmfit
- scikit-learn
- seaborn
- matplotlib

## DEMO

Please run:
- `sh test_cosmo_stream.sh "ecommerce_www2022"`
- `sh test_cosmo_stream.sh "vod_www2022"`
- `sh test_cosmo_stream.sh "sweets_www2022"`
- `sh test_cosmo_stream.sh "facilities_www2022"`

which start the grid search to obtain optimal numbers of latent species/components ($k_y$) and seasonal patterns ($k_z$), initialize the fist regime with the 3-year data, and then start stream forecasting.

The outputs will automatically appear in `out` directory.
