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

    #Splitting the data into train and test sets to check the model 
    train = mod_df.iloc[:len(mod_df)-10]
    test = mod_df.iloc[len(mod_df)-10:] 

    #Training model using train data and train exo data
    train_model = SARIMAX(train[f'{selected_country}_{var}'], 
                    order = best_order, 
                    seasonal_order =best_s_order)

    train_result = train_model.fit()
    # print(train_result.summary())

    #Getting predictions for the text dataset
    start = len(train)
    end = len(train) + len(test) - 1
    predictions = train_result.predict(start, end).rename("Predictions")

    #PLotting predictions and actual values
    predictions.plot(legend = True)
    test[f'{selected_country}_{var}'].plot(legend = True)
    plt.show()

    #Creating the best SARIMAX model using the best values of p,d,q
    model =  SARIMAX(mod_df[f'{selected_country}_{var}'], 
                            order = best_order, 
                            seasonal_order =best_s_order)
    result = model.fit()

    #Forecasting chosen variable based on the number of years given by user
    forecast = result.get_forecast(steps=forecasting_length)
    var_forecast = forecast.predicted_mean

    #Getting graphs for each variable forecast
    fig = go.Figure()

    # Actual variable plotting
    fig.add_trace(go.Scatter(
        x=mod_df.index,
        y=mod_df[f'{selected_country}_{var}'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue'),
        marker=dict(size=6),
        hovertemplate='<b>Year: %{x|%Y}<br>Actual: %{y:.2f}<extra></extra></b>' #Text to be shown while hovering over line
    ))

    # Forecasted variable plotting
    future_index = pd.date_range(start=mod_df.index[-1] + pd.DateOffset(years=1), periods=forecasting_length, freq='Y')
    fig.add_trace(go.Scatter(
        x=future_index,
        y=var_forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green', dash='dash'),
        marker=dict(size=6),
        hovertemplate='<b>Year: %{x|%Y}<br>Forecast: %{y:.2f}<extra></extra></b>' #Text to be shown while hovering over line
    ))

    #Predictions for test data plotting
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions,
        mode='lines+markers',
        name='Test Prediction',
        line=dict(color='red'),
        marker=dict(size=6),
        hovertemplate='<b>Year: %{x|%Y}<br>Test Predicted: %{y:.2f}<extra></extra></b>' #Text to be shown while hovering over line
    ))

    #Formatting the overall layout of the graph
    fig.update_layout(
        title=f"Forecast for {var.upper()}",
        xaxis_title='Year',
        yaxis_title=var.upper(),
        height=350,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(size=10)),
        
        xaxis=dict(
            title_font=dict(size=17, color='black', family='Calibri'),
            tickfont=dict(size=15, color='black', family='Calibri')
        ),
        yaxis=dict(
            title_font=dict(size=17, color='black', family='Calibri'),
            tickfont=dict(size=15, color='black', family='Calibri')
        )
    )

    return var_forecast, fig 