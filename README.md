# Predicting-Late-Delivery-In-Supply-Chains

Introduction:

Monitoring the delivery performance and accurately predicting any delays in deliveries could be one of the key performance indicators for an enterprise to improve customer
retention and ensure service satisfaction. The existing delivery performance assessment system can help to monitor the current delivery performance of sales orders and stock
transport orders. However, for enterprises that are keen to predict the delays in advance for planned deliveries, this proposed delivery risk prediction model can be a useful
tool. The predictive model proposed here uses historical supply chain data gathered for a specific period as a source. The model developed can successfully predict any possible
delays before hand which is a useful tool for enterprises.

This repository contains an automated python script which can be used for creating a predictive model for late delivery risk assessment. The python script recieves input data
from the user, then proceeds with feature engineering and feature selection before building several prediction models using different Machine Learning Algorithms. After 
building all models, the script finds the best model and saves it as a pickle file. The repository also includes a desktop application using which any user can build models
by interacting with the Graphical User Interface.


Contents of this Repository:

1. DesktopAppToBuildDelayPredictionModel
 
The folder DesktopAppToBuildDelayPredictionModel contains a Desktop Application which can recieve inputs from user about the dataset and hyperparamenters. This application
can then run a python script which build several prediction models using various Machine Learning algorithms such as Random Forest, CatBoost, LightGBM, etc.. and then will 
find the model with best performance metric and will output the results to user.

This desktop app is aimed for any firms to build a model for themselves if they have a dataset of their supply chain delivery performance.

2. Automated Python Script.

The .py file residing in this folder is a script using which the above standalone can build a best possible model for end user. For the time being, the location of this 
file has to be given as input to the Desktop Application. After this .py script is completely optimized, this file can be integrated to the Desktop Application. But for the
time being, this python script is open to further optimization.


What an Enterprise need to use this Prediction Model?

a) A dataset of delivery details of orders made in the past few years. The dataset should contain a column which says whether delivery of an order was delayed or not. 
Enterprises can create this coulmn themselves if actual delivery dates are available. 
b) Python should be installed in the Machine.

