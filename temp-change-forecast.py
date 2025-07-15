#Importing all the necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

#Getting the function from utils.py
from utils import ind_var_forecast

#Seeting the layout of streamlit to wide
st.set_page_config(layout="wide")

#Getting the country df to be used for user input and forecasting 
countries_df = pd.read_csv(r'country_list.csv')

#Getting the list of countries from the dataframe
countries = sorted(list(set(countries_df['Country Name'].to_list())))

#Asking the user to select the country that they want the forecast for
selected_country = st.sidebar.selectbox('Select a country:', countries)

#Getting the required dataset with data from all countries 
df = pd.read_csv('combined_df.csv')

#Getting only the data for the selected country and adding the year column
df_selected_0 = df.filter(like=selected_country, axis = 1)
df_selected = pd.concat([df_selected_0, df[['Year']]], axis = 1)
df_selected['Year'] = pd.to_datetime(df_selected['Year'])

#Required cleaning before forecasting
df_selected = df_selected.replace('..', np.nan)

#Getting all variables to be used for forecasting
selected_cols = list(set(df_selected.columns.to_list()) )

# print(selected_cols)

#Setting the index to be the year for easy plotting
selected_df = df_selected.set_index('Year')
# print(selected_df.columns)

#Getting all required columns except the year 
selected_cols_no_yr = [f'{selected_country}_co2',f'{selected_country}_meth', f'{selected_country}_nox',f'{selected_country}_pop', f'{selected_country}_temp']

#print(selected_cols_no_yr)

#Getting the final dataframe for forecasting
selected_df_new = selected_df[selected_cols_no_yr]
selected_df_new[selected_cols_no_yr] = selected_df_new[selected_cols_no_yr].astype(float)
selected_final = selected_df_new[[f'{selected_country}_co2',f'{selected_country}_meth', f'{selected_country}_nox', f'{selected_country}_temp']][30:61]


selected_final1 = selected_final.reset_index()

#Mapping the display names (provided to users) to code names in the data
exog_display_map = {
    'Carbon Dioxide': 'co2',
    'Methane': 'meth',
    'Nitrous Oxide': 'nox',
}

#Letting the user choose the external variables to be used for forecasting
selected_external_vars = st.sidebar.multiselect(
    'Select external variables to use for temperature change forecasting:',
    options=list(exog_display_map.keys()),
    default=list(exog_display_map.keys())
)

#Letting the users choose the numbers of years to forecast
future_values = st.sidebar.slider("Select the number of years to forecast:", max_value=24, min_value=2)


#Once the submit button is pressed
if st.sidebar.button("Submit"):

    #Showing custom Loading GIF and message using a placeholder
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
        <div style='text-align: center;'>
            <p style='font-size:30px;'>Training the SARIMAX model and forecasting temperature change...</p>
            <img src='https://i.imgur.com/KD1eZ9m.gif' width='300'>
        </div>
    """, unsafe_allow_html=True)

    #Convert to code variables based on input from user
    selected_exogs = []
    for name in selected_external_vars:
        selected_exogs.append(exog_display_map[name])

    #Need to work in cases where user do not select any external variables
    if len(selected_exogs) == 0:

        #Getting only the temperature values and year
        temp_initial = selected_final1[[f'{selected_country}_temp', 'Year']].set_index('Year')

        #Using auto_arima to get the best values for p, d, q, for temperature
        model_testing = auto_arima(
        temp_initial,
        start_p = 1, start_q = 1,
                            max_p = 3, max_q = 3, m = 12,
                            start_P = 0, seasonal = True,
                            d = None, D = 1, trace = True,
                            error_action ='ignore',   # we don't want to know if an order does not work
                            suppress_warnings = True,  # we don't want convergence warnings
                            stepwise = True
        )

        best_order = model_testing.order
        best_s_order = model_testing.seasonal_order

        # Splitting the temperature data into train and test sets
        train = temp_initial.iloc[:len(temp_initial)-10]
        test = temp_initial.iloc[len(temp_initial)-10:]

        #Using the train data to train a model to check how good the model is
        model_training = SARIMAX(train, 
                    order = best_order, 
                    seasonal_order =best_s_order)

        train_result = model_training.fit()
        # print(train_result.summary())

        #Getting predictions for the text dataset
        start = len(train)
        end = len(train) + len(test) - 1
        predictions = train_result.predict(start, end).rename("Predictions")
        

        #Creating the best SARIMAX model using the best values of p,d,q (no exo variables used since the user did not choose any)
        model = SARIMAX(
            temp_initial,                  
            order=best_order,
            seasonal_order=best_s_order
        )
        results = model.fit()

        #Forecasting temperature change based on the number of years given by user
        forecast = results.get_forecast(steps=future_values, exog=future_values)
        temp_forecast = forecast.predicted_mean
        # conf_int = forecast.conf_int()
        

    #IF external variables are selected, we need to first get forecast for those and then use them for our temp change forecast
    else:

        # Preparing initial exo variables and future exo variables using the function ind_var_forecast defined in util.py
        initial_exo_cols = [f'{selected_country}_{var}' for var in selected_exogs]
        initial_exo = selected_final1[initial_exo_cols + ['Year']].set_index('Year')

        future_exog = pd.DataFrame()

        exo_figs = []

        #Mapping dictionary for variable with actual values
        exo_title_map = {
            'co2': 'CO₂ emissions (metric tons per capita)',
            'meth': 'Methane emissions (metric tons of CO₂ equivalent per capita)',
            'nox': 'Nitrous oxide emissions (metric tons of CO₂ equivalent per capita)'
        }

        #Getting the forecasts for exo variables along with the graphs
        for var in selected_exogs:
            forecast_series, exo_fig = ind_var_forecast(selected_final1, selected_country, var, forecasting_length=future_values)
            future_exog[var] = forecast_series
            title = exo_title_map.get(var, var.upper())

            y_axis = title.split('(')[0]
            exo_fig.update_layout(
                title=f'Forecast for {title}',
                yaxis=dict(
                    title=dict(
                        text=f'{y_axis}',
                        font=dict(size=17, color='black', family='Calibri', **{'weight': 'bold'})
                    ),
                    tickfont=dict(size=15, color='black', family='Calibri')
                )
            )
            exo_figs.append((title, exo_fig))

        #Getting the initial temperature values 
        temp_initial = selected_final1[[f'{selected_country}_temp', 'Year']].set_index('Year')

        #Getting the initial exo variables with year as index
        initial_exo_cols = [f'{selected_country}_{var}' for var in selected_exogs]
        initial_exo = selected_final1[initial_exo_cols + ['Year']].set_index('Year')

        #Using auto_arima to get the best values for p, d, q, for temperature with initial exo variables
        model_testing = auto_arima(
            temp_initial,
            exogenous=initial_exo, 
            seasonal=True,             
            stepwise=True,
            max_p=5, max_q=5,
            d=None,                  
            trace=True,
            suppress_warnings=True
        )

        best_order = model_testing.order
        best_s_order = model_testing.seasonal_order

        #Splitting the data into train and test sets to check the model 
        train = temp_initial.iloc[:len(temp_initial)-10]
        test = temp_initial.iloc[len(temp_initial)-10:]

        train_exo = initial_exo.iloc[:len(initial_exo)-10]
        test_exo = initial_exo.iloc[len(initial_exo)-10:]

        #Training model using train data and train exo data
        model_training = SARIMAX(train, 
                    exogenous=train_exo, 
                    order = best_order, 
                    seasonal_order =best_s_order)

        train_result = model_training.fit()
        # print(train_result.summary())

        #Getting predictions for the text dataset
        start = len(train)
        end = len(train) + len(test) - 1
        predictions = train_result.predict(start, end, exog=test_exo).rename("Predictions")

    
        #Creating the best SARIMAX model using the best values of p,d,q (exo variables used)
        model = SARIMAX(
            temp_initial,                
            exog=initial_exo, 
            order=best_order,
            seasonal_order=best_s_order
        )
        results = model.fit()

        #Forecasting temperature change based on the number of years given by user
        forecast = results.get_forecast(steps=len(future_exog), exog=future_exog)
        temp_forecast = forecast.predicted_mean
        # conf_int = forecast.conf_int()


    #Getting the final temp value from inital training data so the temp graph is connected
    final_initial_temp = temp_initial.tail(1).rename(columns={f'{selected_country}_temp' : 'predicted_mean'})
    forecast_final = pd.concat([final_initial_temp, temp_forecast])


    st.subheader("Temperature Change Forecast", divider='grey')
    
    #Using plotly to create interactive graph for temp change forecast

    fig = go.Figure()

    # Actual temperature change plotting
    fig.add_trace(go.Scatter(
        x=temp_initial.index,
        y=temp_initial.iloc[:, 0],
        mode='lines+markers', #Adding both the line and the markers
        name='Actual Temp Change', #Label of the line 
        line=dict(color='blue'), 
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Year: %{x|%Y}<br>Actual Temperature: %{y:.2f} °C<extra></extra></b>' #Text to be shown while hovering over line
    ))


    # Forecasted temperature change plotting
    fig.add_trace(go.Scatter(
        x=forecast_final.index,
        y=forecast_final.iloc[:, 0],
        mode='lines+markers',
        name='Forecasted Temp Change',
        line=dict(color='green', dash='dash'), #Making the line dashed line
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Forecast Year: %{x|%Y}<br>Predicted Temperature: %{y:.2f} °C<extra></extra></b>'
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
        # title="Temperature change forecast", #Removed since the streamlit title is enough 
        xaxis_title="Year",
        yaxis_title="Temperature Change (°C)",
        font=dict(size=10),
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    xaxis=dict(
        title_font=dict(size=17, color='black', family='Calibri', weight='bold'),
        tickfont=dict(size=15, color='black', family='Calibri')
    ),
    yaxis=dict(
        title_font=dict(size=17, color='black', family='Calibri', weight='bold'),
        tickfont=dict(size=15, color='black', family='Calibri')
    ),
    #Adding a dashed red line at the 0 temp value
    shapes=[
        dict(
            type="line",
            xref="paper",  
            yref="y",      
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            line=dict(color="red", width=2, dash="dot")
        )
    ]
    )

    #Removing the 'loading' text and gif once graph is ready to be displayed
    loading_placeholder.empty()

    st.plotly_chart(fig, use_container_width=True)

    #Adding a line divider before shown additional information about the graph
    st.divider() 

    #Showing additonal information about the SARIMAX graph 
    #Number of future years forecasted 
    #Exo variables used 
    st.markdown(f"**Number of years forecasted: {future_values}**")

    if len(selected_exogs) == 0:
        st.markdown(f"**No exogenous variables used.**")
    elif len(selected_exogs) > 0:
        st.markdown(f"**Exogenous variables used:**")
        for i, j in enumerate(selected_external_vars):
            st.markdown(f'**{i+1}. {j}**')

        #Plotting the graphs and forecasts for the exo variables
        st.subheader("Forecasts for Selected External Variables", divider='grey')

        for title, fig in exo_figs:
            st.plotly_chart(fig, use_container_width=True)




