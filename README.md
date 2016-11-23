
Follow along in the code from Steps I-VII 

Here are a few exercises to further explore and understand the dl4j api. 

1. Take a look at the csv files written out. Compare this to the raw data that can be viewed [here](https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data) 

   The format: one time series per file, and a separate file for the labels matched up by name.For example, train/features/0.csv is the features using with the labels file train/labels/0.csv 
   The data here is a univariate time series, we only have one column in the CSV files.
   Furthermore, because we have only one label for each time series, the labels CSV files contain only a single value 
   For more details on importing time series and how to handle time series with different lengths, refer [here](http://deeplearning4j.org/usingrnns#data)

2. Vary the mini batch size and note the effect on the score function in the UI. What happens when a larger mini batch size is used? A smaller? 

3. Vary the learning rate by 100x,10x and 0.1x. Note the effect on the scores in the UI. 

4. Leave out normalization. Does your net learn? Try another normalization technique. 

5. Explore the different updaters and optimization algorithms. 

