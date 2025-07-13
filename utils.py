import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


#Creating a fucntion to forecast the exo variables based on the user input
def ind_var_forecast(df, selected_country, var, forecasting_length):
    mod_df = df[['Year',f'{selected_country}_{var}']].set_index('Year')

    # Fitting an auto_arima function to to get the best values for p, q, d
    stepwise_fit = auto_arima(mod_df[f'{selected_country}_{var}'], start_p = 1, start_q = 1,
                            max_p = 3, max_q = 3, m = 12,
                            start_P = 0, seasonal = True,
                            d = None, D = 1, trace = True,
                            error_action ='ignore',  
                            suppress_warnings = True,  
                            stepwise = True)           

    best_order = stepwise_fit.order
    best_s_order = stepwise_fit.seasonal_order

    # Split data into train / test sets
    train = mod_df.iloc[:len(mod_df)-10]
    test = mod_df.iloc[len(mod_df)-10:] # set one year(12 months) for testing



    # Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set
    

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


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mod_df.index,
        y=mod_df[f'{selected_country}_{var}'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))

    future_index = pd.date_range(start=mod_df.index[-1] + pd.DateOffset(years=1), periods=forecasting_length, freq='Y')
    fig.add_trace(go.Scatter(
        x=future_index,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', dash='dash'),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=f"Forecast for {var.upper()}",
        xaxis_title='Year',
        yaxis_title=var.upper(),
        height=350,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(size=10)),
        
        xaxis=dict(
            title_font=dict(size=17, color='black', family='Calibri', weight='bold'),
            tickfont=dict(size=15, color='black', family='Calibri')
        ),
        yaxis=dict(
            title_font=dict(size=17, color='black', family='Calibri', weight='bold'),
            tickfont=dict(size=15, color='black', family='Calibri')
        )
    )

    return forecast, fig 