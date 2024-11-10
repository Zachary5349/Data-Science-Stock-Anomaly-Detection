# Stock Anomaly Detection

## A fully functional anomaly detector for any stock

### Disclaimer 
This application is for informational purposes only. Stock trading involves risks, and users are
still responsible for their own decisions. 

Welcome to the Stock Anomaly Detector, two programs that can determine anomalies in previous stock data to help
the user find trends and know when a stock reaching a unreasonably high or low value.

## For Developers: Comments are included in the code explaining functions and chunks of code. These will help explain how the program works.

## For Users and Developers: To run the program, many packages are required. Packages include: 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

from sklearn.preprocessing import StandardScaler

import yfinance as yf

from adtk.data import validate_series

from adtk.detector import ThresholdAD


After importing the latest versions of the packages, you should be able to run the code without errors. 

Before running the program, you must go into the code of the program you are trying to run and find the ticker symbol.
After changing this ticker symbol to the desired stock, the program will then run the method(s) for the stock that you entered. 
The program will display graphs for the anomaly detection methods and the outliers will be dots.

## If you find a bug or a possible improvement to this project, please submit an issue in the issues tab above. Thank you!
