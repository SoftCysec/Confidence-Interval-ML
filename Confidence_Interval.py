########
# Data #
########
ESTIMATIONS_FILE = 'estimations.csv'
MEASUREMENTS_FILE = 'measurements.csv'

#########
# MODEL #
#########

# Select which features to use for the model
FEATURES_TO_USE = ['hour', 'estimation']

# Mean and STD of estimation feature after min/max normalization
# These are used for standardization of the estimation feature, which allows for using a sigmoid transformation
# See generate_model_data() for how these values are calculated. These values are calculated on all available data.
ESTIMATION_MEAN = 0.33736
ESTIMATION_STD = 0.11137

##############
# EVALUATION #
##############
DAYS_TO_PLOT = 5
TEST_MRIDS = ['00bffd24-1f0b-586b-b353-4cea5ceaff1b',
'0c0ad75e-7c00-5393-8f79-03a1d11aadd0',
'1f226e09-3f9e-52e4-984c-c0749b9654d0',
'2af8dd36-5065-571f-aa75-038c7496942e',
'3208c09f-88b3-470a-b8c9-ac2f18b028b3',
'46928fb1-083e-520d-b3a5-57b9e1b5e1a2',
'5fdaa414-c47b-5eea-a9b3-71a5c720dd47',
'8ceaaff9-5406-5420-bf57-436f189a96ed',
'985f8460-2e08-5054-bd81-126d544cabd4',
'af895424-9a66-4695-bfa4-7cfaa8eac64f',
'd364be8b-cbee-56b1-8732-28b85f7dea4c',
'e6b10eef-6b87-4ce4-8186-c356b75e7fec',
'ffabbb9c-3d4c-543e-b6e6-80fadf3dea29']

#########
# DEBUG #
#########
DEBUG_LIST_ALL_MSR = False



import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import timedelta
import os
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import datetime
import yaml
import collections
import joblib

st.title('Confidence Intervals Visualized')

st.write('Visualizing the head')
data_dir = os.getcwd()
estimations = pd.read_csv(os.path.join(data_dir,ESTIMATIONS_FILE), index_col=0, parse_dates=True)
measurements = pd.read_csv(os.path.join(data_dir,MEASUREMENTS_FILE), index_col=0, parse_dates=True)

# How wrong are the estimations compared to the measurements at each point in time
errors = estimations - measurements

st.write((estimations.head()))



st.header('Error distribution per time-of-day (hour)')
 
"Useful to see how the errors are distributed, and how this changes for each of the 24 hours of a day."


def get_mean_and_std_for_hour(data, hour):
    data_for_single_hour = data[(data.index.hour >= hour) & (data.index.hour < hour+1)]
    # Get all data points for that single hour for all MSRs into a single array
    # We loose information about the individual MRIDs
    distr = data_for_single_hour.to_numpy().flatten()
    
    # Return the mean and STD of all those data points
    return distr, distr.mean(), distr.std()

err_means = []
err_stds = []
for hour in range(24):
    _, mean, std = get_mean_and_std_for_hour(errors, hour)
    err_means.append(mean)
    err_stds.append(std)
    
data_1 = plt.figure(figsize=(10,6))
plt.plot(err_means, label="Mean")
plt.plot(err_stds, label="STD")
plt.xlabel("Hour of day")
plt.ylabel("Predictions - Measurements")
plt.title("Mean and STD of errors per hour")
plt.xticks(range(24))
plt.legend()

st.write(data_1)

st.header('Feature transformations')
"Linearlization of features by applying transformations to each feature. This is very important for linear regression."


# Feature transformations
def transform_hour(hour):
    return np.sin((hour*np.pi*2)-np.pi/2)

def transform_log(x):
    return np.log(x+0.00001)

def standardize(x):
    return (x - x.mean())/(x.std())

def sigmoid(x):
    return 1/(1 + np.exp(-x))



def apply_all_transformations(data_input):
    data = data_input.copy()
    data['hour'] = transform_hour(data['hour'])
    data['estimation'] = sigmoid((data['estimation'] - ESTIMATION_MEAN)/ESTIMATION_STD)
    return data

st.header('Creating the model and testing')

def fit_linreg_model(data):
    # Train the linear regression model 
    return LinearRegression().fit(data[FEATURES_TO_USE], data[['mean', 'STD']])


# Returns dataframe with input for model plus the output labels, and the scaler used to min-max normalize the input
def generate_model_data(measured, estimated, do_transformations=True):
    
    # Generates the mean and STD label for a given MRID-hour combination
    def get_label_for_mrid_and_hour(data, mrid, hour):
        distr = data[mrid][data.index.hour == hour]
        #distr.plot.hist()
        #plt.show()
        return distr.mean(), distr.std()
    
    differences = estimated - measured
    
    # Build dataframe as list of dicts
    # Note that this is quite inefficient and that it can be done faster using df.melt and df.pivot
    rows_list = []
    for mrid in tqdm(estimated.columns, desc = 'Building data for MRIDs'):
        
        for timestamp, estimation in estimated[mrid].items():
            mean, std = get_label_for_mrid_and_hour(differences, mrid, timestamp.hour)
            row_dict = {
                "MRID": mrid,
                "hour": timestamp.hour,
                "estimation": estimation,
                "mean": mean,
                "STD": std
            }
            
            rows_list.append(row_dict)
            
    # Normalize features
    scaler = MinMaxScaler()
    return_df = pd.DataFrame(rows_list)
    return_df[FEATURES_TO_USE] = scaler.fit_transform(return_df[FEATURES_TO_USE])
    
    # Code that was used to calculate the ESTIMATION_MEAN and ESTIMATION_STD
    #print("Estimation mean value:", return_df['estimation'].mean())
    #print("Estimation STD value:", return_df['estimation'].std())
    
    # Apply transformations to the features
    if do_transformations:
        return_df = apply_all_transformations(return_df)
    
    return return_df, scaler




# Generate train data, keep out TEST_MRIDs for evaluation
train_measurements = measurements.drop(TEST_MRIDS,axis=1)
train_estimations = estimations.drop(TEST_MRIDS,axis=1)

# Generate data for model
print("Generating train dataframe")
train_df, scaler = generate_model_data(train_measurements, train_estimations)

# Train model
with st.spinner('Training LinReg model'):
    linreg_model = fit_linreg_model(train_df)



def plot_single_mrid(df_single_mrid, mrid, days_to_plot=31):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=mrid, xaxis_title="Time", yaxis_title="Load")
    
    # Lower bound
    fig.add_trace(go.Scatter(x=df_single_mrid.index[:24*4*days_to_plot],
                             y=df_single_mrid['lower_bound'].iloc[:24*4*days_to_plot],
                             fill=None,
                             mode='lines',
                             line_color='orange',
                             name="Lower bound",
                             showlegend=False,
                             legendgroup="bound"
                            )
                 )
    # Upper bound
    fig.add_trace(go.Scatter(x=df_single_mrid.index[:24*4*days_to_plot],
                             y=df_single_mrid['upper_bound'].iloc[:24*4*days_to_plot],
                             fill='tonexty', # fill area between lower and upper bound
                             opacity=0.1,
                             mode='lines',
                             line_color='orange',
                             name="Upper/lower bound",
                             legendgroup="bound"
                            )
                 )
    # Estimations
    fig.add_trace(go.Scatter(x=df_single_mrid.index[:24*4*days_to_plot],
                             y=df_single_mrid['estimation'].iloc[:24*4*days_to_plot],
                             mode="markers",
                             marker=dict(
                                 color='blue',
                                 size=5,
                             ),
                             name="Estimations"
                            )
                 )
    # LS Measurements background line (All)
    fig.add_trace(go.Scatter(x=df_single_mrid.index[:24*4*days_to_plot],
                             y=df_single_mrid['measurement'].iloc[:24*4*days_to_plot],
                             mode="lines",
                             line_color='grey',
                             opacity=0.75,
                             name="Measurement line"
                            )
                 )
    # LS Measurements (All)
    fig.add_trace(go.Scatter(x=df_single_mrid.index[:24*4*days_to_plot],
                             y=df_single_mrid['measurement'].iloc[:24*4*days_to_plot],
                             mode="markers",
                             marker=dict(
                                 color='green',
                                 size=5,
                             ),
                             name="Measurements"
                            )
                 )
    # LS Measurements outside bounds
    fig.add_trace(
        go.Scatter(
            x = df_single_mrid['measurement'].iloc[:24*4*days_to_plot][(
                (df_single_mrid['measurement']>df_single_mrid['upper_bound'])|
                (df_single_mrid['measurement']<df_single_mrid['lower_bound'])
                )].index,
            y = df_single_mrid['measurement'].iloc[:24*4*days_to_plot][(
                (df_single_mrid['measurement']>df_single_mrid['upper_bound'])|
                (df_single_mrid['measurement']<df_single_mrid['lower_bound'])
                )],
            mode = "markers",
            marker = dict(
                color='red',
                size=5,
                ),
            name = "Measurements outside bounds"
            )
        )
    
    st.write(fig)
    
    


st.subheader('Test a single MRID using given mode')
# Test a single MRID using given model
def test_single_mrid_model(mrid, model, do_plot=True, do_display=False, days_to_plot=31, symmetric=False):
    # Combine measurements and estimations for single MRID into one Dataframe
    df_single_mrid = pd.DataFrame().assign(measurement=measurements[mrid], estimation=estimations[mrid])
    
    # Add column with features used by the model
    df_single_mrid['hour'] = df_single_mrid.index.hour
    
    # Normalize
    df_single_mrid_normalized = df_single_mrid.copy()
    df_single_mrid_normalized[FEATURES_TO_USE] = scaler.transform(df_single_mrid_normalized[FEATURES_TO_USE])
    
    # Transform features
    df_single_mrid_normalized = apply_all_transformations(df_single_mrid_normalized)
    
    # Make predictions for mean and STD using the model
    predictions = model.predict(df_single_mrid_normalized[FEATURES_TO_USE])
    
    # Add predictions to dataframe (transpose is needed to align with dataframe)
    df_single_mrid['mean'], df_single_mrid['STD'] = predictions.T
    
    # Calculate confidence interval using estimation, mean and std
    # upper bound = (est - mean) + std*3
    # upper bound = (est - mean) - std*3
    if symmetric:
        df_single_mrid['upper_bound'] = (df_single_mrid['estimation']) + df_single_mrid['STD']*3
        df_single_mrid['lower_bound'] = (df_single_mrid['estimation']) - df_single_mrid['STD']*3
    else:
        df_single_mrid['upper_bound'] = (df_single_mrid['estimation'] - df_single_mrid['mean']) + df_single_mrid['STD']*3
        df_single_mrid['lower_bound'] = (df_single_mrid['estimation'] - df_single_mrid['mean']) - df_single_mrid['STD']*3
    
    # Mark if points fall outside the confidence interval
    df_single_mrid['outside_interval'] = (
        (df_single_mrid.measurement > df_single_mrid.upper_bound) | 
        (df_single_mrid.measurement < df_single_mrid.lower_bound)
    )
    
    if do_plot:
        plot_single_mrid(df_single_mrid, mrid, days_to_plot=days_to_plot)
        
    return df_single_mrid



# Plot held out test set
for mrid in TEST_MRIDS:
    st.write(test_single_mrid_model(mrid, linreg_model, do_plot=True, days_to_plot=DAYS_TO_PLOT))