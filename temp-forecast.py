#Importing all the necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

#Getting the function from utils.py
from utils import ind_var_forecast

#Seeting the layout of streamlit to wide
st.set_page_config(layout="wide")

#Getting the country list to be used for user input and forecasting 
df0 = pd.read_csv(r'countries_df.csv')

countries = sorted(list(set(df0['countries'].to_list())))

#Asking user to select the country that they want the forecast for
selected_country = st.sidebar.selectbox('Select a country:', countries)


#getting the dataset with data from each country 
df = pd.read_csv('combined_df.csv')


df_selected_0 = df.filter(like=selected_country, axis = 1)
df_selected = pd.concat([df_selected_0, df[['Year']]], axis = 1)
df_selected['Year'] = pd.to_datetime(df_selected['Year'])

df_selected = df_selected.replace('..', np.nan)

selected_cols = list(set(df_selected.columns.to_list()) )
# selected_cols

print(selected_cols)

selected_df = df_selected.set_index('Year')
print(selected_df.columns)

selected_cols_no_yr = [f'{selected_country}_co2',f'{selected_country}_meth', f'{selected_country}_nox',f'{selected_country}_pop', f'{selected_country}_temp', f'{selected_country}_land_temp']

print(selected_cols_no_yr)

selected_df_new = selected_df[selected_cols_no_yr]

selected_df_new[selected_cols_no_yr] = selected_df_new[selected_cols_no_yr].astype(float)

# print(selected_df_new.head())




selected_final = selected_df_new[[f'{selected_country}_co2',f'{selected_country}_meth', f'{selected_country}_nox', f'{selected_country}_temp', f'{selected_country}_land_temp']][30:61]


selected_final1= selected_final.reset_index()



# co2_forecast = ind_var_forecast(selected_final1, 'co2')
# meth_forecast = ind_var_forecast(selected_final1, 'meth')
# nox_forecast = ind_var_forecast(selected_final1, 'nox')
# pop_forecast = ind_var_forecast(selected_final1, 'pop')

# future_exog = pd.DataFrame({
#     'co2': co2_forecast,
#     'meth': meth_forecast,
#     'nox': nox_forecast,
#     'pop': pop_forecast
# })


# Mapping from display names to code names
exog_display_map = {
    'Carbon Dioxide': 'co2',
    'Methane': 'meth',
    'Nitrous Oxide': 'nox',
    # 'Population': 'pop'
}


selected_display = st.sidebar.multiselect(
    'Select external variables to use for temperature prediction:',
    options=list(exog_display_map.keys()),
    default=list(exog_display_map.keys())
)



future_values = st.sidebar.slider("Select the number of years to forecast:", max_value=24, min_value=2)


if st.sidebar.button("Submit"):

        # Show loading GIF and message using a placeholder
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
        <div style='text-align: center;'>
            <p style='font-size:30px;'>Training SARIMAX model and forecasting temperature change...</p>
            <img src='https://i.imgur.com/KD1eZ9m.gif' width='350'>
        </div>
    """, unsafe_allow_html=True)




    # with st.spinner("Training model and forecasting temperature..."):
        # Convert to backend codes
    selected_exogs = [exog_display_map[name] for name in selected_display]

    # Handle empty selection
    ##Need to change this so there is forecatsing without exo variables as well
    if len(selected_exogs) == 0:
        temp_initial = selected_final1[[f'{selected_country}_temp', 'Year']].set_index('Year')
        stepwise_model = auto_arima(
        temp_initial,
        start_p = 1, start_q = 1,
                            max_p = 3, max_q = 3, m = 12,
                            start_P = 0, seasonal = True,
                            d = None, D = 1, trace = True,
                            error_action ='ignore',   # we don't want to know if an order does not work
                            suppress_warnings = True,  # we don't want convergence warnings
                            stepwise = True
        )

        # print(stepwise_model.summary())

        best_order = stepwise_model.order
        best_s_order = stepwise_model.seasonal_order

        # print(best_order, best_s_order)

        

        model = SARIMAX(
            temp_initial,                  # temperature from 1990–2020
            # exog=initial_exo,  # CO2, CH4, NOx, population for 1990–2020
            order=best_order,
            seasonal_order=best_s_order
        )
        results = model.fit()

        forecast = results.get_forecast(steps=future_values, exog=future_values)
        temp_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        # st.warning("Please select at least one external variable to proceed.")
        # st.stop()
    else: 
        # Prepare initial_exo and future_exog
        initial_exo_cols = [f'{selected_country}_{var}' for var in selected_exogs]
        initial_exo = selected_final1[initial_exo_cols + ['Year']].set_index('Year')

        future_exog = pd.DataFrame({var: ind_var_forecast(selected_final1, selected_country, var, forecasting_length=future_values) for var in selected_exogs})


        # st.line_chart(future_exog[['co2', 'meth', 'nox']])
        # print(co2_forecast)

        temp_initial = selected_final1[[f'{selected_country}_temp', 'Year']].set_index('Year')
        # initial_exo = selected_final1[[f'{selected_country}_co2',f'{selected_country}_meth', f'{selected_country}_nox',f'{selected_country}_pop',"Year"]].set_index('Year')
        # print(initial_exo)

        initial_exo_cols = [f'{selected_country}_{var}' for var in selected_exogs]

        initial_exo = selected_final1[initial_exo_cols + ['Year']].set_index('Year')

        stepwise_model = auto_arima(
            temp_initial,
            exogenous=initial_exo,     # your exog vars
            seasonal=True,             # set to True if seasonality exists
            stepwise=True,
            max_p=5, max_q=5,
            d=None,                     # Let auto_arima decide
            trace=True,
            suppress_warnings=True
        )

        # print(stepwise_model.summary())

        best_order = stepwise_model.order
        best_s_order = stepwise_model.seasonal_order

        # print(best_order, best_s_order)

        

        model = SARIMAX(
            temp_initial,                  # temperature from 1990–2020
            exog=initial_exo,  # CO2, CH4, NOx, population for 1990–2020
            order=best_order,
            seasonal_order=best_s_order
        )
        results = model.fit()

        forecast = results.get_forecast(steps=len(future_exog), exog=future_exog)
        temp_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()

    # st.line_chart(temp_initial)
    # st.line_chart(temp_forecast)

    final_initial_temp = temp_initial.tail(1).rename(columns={f'{selected_country}_temp' : 'predicted_mean'})
    forecast_final = pd.concat([final_initial_temp, temp_forecast])
    print(forecast_final)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(forecast_final, label ="Forecasted", color = 'red', marker = 'o', markersize = 2.5, linestyle = '--', linewidth=0.8)
    ax.plot(temp_initial, label='Actual', marker = 'o', markersize = 2.5, linestyle='--', linewidth=0.8)


    st.subheader("Temperature Change Forecast", divider='grey')
    

    #Using matplotlib

    # ax.set_title("Temperature change forecast", fontsize = 8)
    # ax.set_xlabel("Year", fontsize = 7)
    # ax.set_ylabel("Temperature Change", fontsize = 7)
    # ax.tick_params(axis='both', labelsize=6)
    # ax.legend(fontsize=6)
    # ax.grid()

    # loading_placeholder.empty()
    # st.pyplot(fig)

    import plotly.graph_objects as go

    fig = go.Figure()

    # Actual temperature
    fig.add_trace(go.Scatter(
        x=temp_initial.index,
        y=temp_initial.iloc[:, 0],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Year: %{x|%Y}<br>Actual Temperature: %{y:.2f} °C<extra></extra></b>'
    ))

    # Forecasted temperature
    fig.add_trace(go.Scatter(
        x=forecast_final.index,
        y=forecast_final.iloc[:, 0],
        mode='lines+markers',
        name='Forecasted',
        line=dict(color='green', dash='dash'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Forecast Year: %{x|%Y}<br>Predicted Temperature: %{y:.2f} °C<extra></extra></b>'
    ))

    fig.update_layout(
        # title="Temperature change forecast",
        xaxis_title="Year",
        yaxis_title="Temperature Change",
        font=dict(size=10),
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
            xaxis=dict(
        title_font=dict(size=17, color='black', family='Calibri', weight='bold'),
        tickfont=dict(size=15, color='black', family='Calibri')
    ),
    yaxis=dict(
        title_font=dict(size=17, color='black', family='Calibri', weight='bold'),
        tickfont=dict(size=15, color='black', family='Calibri')
    ),
    shapes=[
        dict(
            type="line",
            xref="paper",  # use entire x-axis (0 to 1 in paper coordinates)
            yref="y",      # actual y-axis values
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            line=dict(color="red", width=2, dash="dot")
        )
    ]
    )

    loading_placeholder.empty()
    st.plotly_chart(fig, use_container_width=True)

    st.divider() 

    st.markdown(f"**Number of years forecasted: {future_values}**")
    if len(selected_exogs) == 0:
        st.markdown(f"**No exogenous variables used.**")
    elif len(selected_exogs) > 0:
        st.markdown(f"**Exogenous variables used:**")
        for i, j in enumerate(selected_display):
            st.markdown(f'**{i+1}. {j}**')



    #show forecasted graphs for exo vars?

