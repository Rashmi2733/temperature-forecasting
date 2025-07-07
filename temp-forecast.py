import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")

from utils import ind_var_forecast

df0 = pd.read_csv(r'countries_df.csv')

countries = sorted(list(set(df0['countries'].to_list())))

selected_country = st.sidebar.selectbox('Select a country:', countries)

# selected_var = st.sidebar.selectbox("Select variable to display:", ['Carbon Dioxide', 'Methane', 'Population', 'Nitrous Oxide'])

# selected_var = st.selectbox("Select variable to display:", ['co2', 'meth', 'pop', 'nox'])

# final_var = ""
# if selected_var == 'Carbon Dioxide':
#     final_var = "co2"
# elif selected_var == 'Population':
#     final_var = "pop"
# elif selected_var == 'Nitrous Oxide':
#     final_var = "nox"
# elif selected_var == 'Methane':
#     final_var = "meth"


df = pd.read_csv('combined_df.csv')

# print(df.shape)


# selected_country = st.text_input('Select a country:')
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




selected_final = selected_df_new[[f'{selected_country}_co2',f'{selected_country}_meth', f'{selected_country}_nox',f'{selected_country}_pop', f'{selected_country}_temp', f'{selected_country}_land_temp']][30:61]

# if selected_country and selected_var:
#     st.line_chart(selected_final[[f'{selected_country}_{final_var}']])


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
    'Population': 'pop'
}


selected_display = st.sidebar.multiselect(
    'Select external variables to use for temperature prediction:',
    options=list(exog_display_map.keys()),
    default=list(exog_display_map.keys())
)

future_values = st.sidebar.slider("Select the number of years to forecast:", max_value=24, min_value=2)


if st.sidebar.button("Submit"):
    with st.spinner("Training model and forecasting temperature..."):
        # Convert to backend codes
        selected_exogs = [exog_display_map[name] for name in selected_display]

        # Handle empty selection
        ##Need to change this so there is forecatsing without exo variables as well
        if len(selected_exogs) == 0:
            st.warning("Please select at least one external variable to proceed.")
            st.stop()

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

        from statsmodels.tsa.statespace.sarimax import SARIMAX

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

        fig, ax = plt.subplots()
        ax.plot(temp_initial, label='Actual')
        ax.plot(temp_forecast, label ="Forecasted")
        ax.set_title("Temp different forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature")
        ax.legend()

        st.pyplot(fig)


        #show forecasted graphs foe exo vars?

