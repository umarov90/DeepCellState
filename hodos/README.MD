This page is for comparison to methods by Hodos et al, DNPP and FaLRTC. 
The accompanying data can be downloaded from [here](https://drive.google.com/file/d/13uQLtbX4HxD-0nLzOuE7UfOjueo6_xMR/view?usp=sharing), which includes the two methods themselves.

The models trained for cross-validation using our data can be downloaded [here](https://drive.google.com/file/d/1iEWsf2lmrRreOn07SsdSCbm0dEKpHflN/view?usp=sharing). 
The models trained for cross-validation using preprocessed data from study by Hodos et al can be downloaded [here](https://drive.google.com/file/d/1WWUl0ufd3rchONCf7UuFaABVpAMXnMtK/view?usp=sharing).

**10_folds_our_data.py** and **10_folds_their_data.py** create testing folds and prepare data for DNPP and FaLRTC. 
The output data by the competitor methods is used by **10_folds_our_data_eval.py** and **10_folds_their_data_eval.py** to make the final evaluation. 

The complete results of running these scripts, competitor methods and evaluation results are present in the downloaded data file (_our_data_, _their_data_). 