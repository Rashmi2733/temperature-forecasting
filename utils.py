import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")



def ind_var_forecast(df, selected_country, var, forecasting_length):
    mod_df = df[['Year',f'{selected_country}_{var}']].set_index('Year')

    # Fit auto_arima function to AirPassengers dataset
    stepwise_fit = auto_arima(mod_df[f'{selected_country}_{var}'], start_p = 1, start_q = 1,
                            max_p = 3, max_q = 3, m = 12,
                            start_P = 0, seasonal = True,
                            d = None, D = 1, trace = True,
                            error_action ='ignore',   # we don't want to know if an order does not work
                            suppress_warnings = True,  # we don't want convergence warnings
                            stepwise = True)           # set to stepwise

    # To print the summary
    best_order = stepwise_fit.order
    best_s_order = stepwise_fit.seasonal_order

    # print(best_order, best_s_order)

    # Split data into train / test sets
    train = mod_df.iloc[:len(mod_df)-10]
    test = mod_df.iloc[len(mod_df)-10:] # set one year(12 months) for testing



    # Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(train[f'{selected_country}_{var}'], 
                    order = best_order, 
                    seasonal_order =best_s_order)

    result = model.fit()
    print(result.summary())


    start = len(train)
    end = len(train) + len(test) - 1

    # Predictions for one-year against the test set
    predictions = result.predict(start, end,
                                typ = 'levels').rename("Predictions")

    # plot predictions and actual values
    predictions.plot(legend = True)
    test[f'{selected_country}_{var}'].plot(legend = True)
    plt.show()

    # Train the model on the full dataset
    model = model = SARIMAX(mod_df[f'{selected_country}_{var}'], 
                            order = best_order, 
                            seasonal_order =best_s_order)
    result = model.fit()

    # Forecast for the next year
    forecast = result.predict(start = len(mod_df), 
                            end = (len(mod_df)-1) + forecasting_length, 
                            typ = 'levels').rename('Forecast')

    # Plot the forecast values
    # mod_df[f'{selected_country}_{var}'].plot(figsize = (12, 5), legend = True)
    # forecast.plot(legend = True)

    return forecast 