import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import joblib


st.title('Pedestrian and Cyclist Safety Prediction in New York City')
st.text('This Application Shows The Risk of Your Location In New York City')
Month = st.slider("Enter Month(from: 1-12):",1,12)
Hour = st.slider("Enter Hour(from: 0-23):",0,23)
Longitude = st.number_input("Enter Longitude (Round with 3 Decimal Places):",min_value=-74.255, max_value=-73.700, format="%.3f")
Latitude = st.number_input("Enter Latitude (Round with 3 Decimal Places):",min_value=40.495, max_value=40.915, format="%.3f")
Hourlyprcp = st.number_input("Enter Precipitation (inches to hundredths),For HourlyParcipitation should select from following range numbers:  Trace - Less than 0.01 inches, Light - 0.01 to 0.10 inches, Light to Moderate - 0.11 to 0.30 inches, Moderate - 0.31 to 1.00 inches, Moderate to Heavy - 1.01 to 2.00 inches, Heavy - 2.01 to 4.00 inches,Very Heavy - 4.01 to 8.00 inches, Extreme - More than 8.00 inches:")




#x=pd.read_csv('myapp_x.csv')
y=pd.read_csv('myapp_y.csv')
z=joblib.load('forest.joblib')




def Risk_of_Location1(month, hour, lat, long, hourlyprcp):
    # Check if the location is in the first dataset and not in the second dataset
  #  if ((x['LONGITUDE'] == long) & (x['LATITUDE'] == lat)).any() and not ((y['LONGITUDE'] == long) & (y['LATITUDE'] == lat)).any():
        #return 'Medium Risk'
    
    # If the location is in the second dataset, use the Random Forest model
    if ((y['LONGITUDE'] == long) & (y['LATITUDE'] == lat)).any():
        prediction = z.predict([[month, hour, long, lat, hourlyprcp]])  
        if prediction == 0:
            return 'High Risk'
        elif prediction == 1:
            return 'Very High Risk'
    
    # If the location is not in either dataset
    else:
        return 'Low  or Medium Risk'
    


if st.button("Risk Category"):
    result = Risk_of_Location1(Month, Hour,Latitude, Longitude, Hourlyprcp)
    st.write("Result:", result)
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

locations = [{"latitude": Latitude, "longitude": Longitude}]

for location in locations:
    lat, lon = location["latitude"], location["longitude"]
    risk_level = Risk_of_Location1(Month, Hour,Latitude, Longitude, Hourlyprcp)
    
    # Define marker colors based on risk levels
    marker_colors = {
        "Low Risk": "green",
        "Medium Risk": "orange",
        "High Risk": "purple",
        "Very High Risk": "red",
    }
    
    marker_color = marker_colors.get(risk_level, "blue")
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.7,
        popup=f"Risk Level: {risk_level}",
    ).add_to(nyc_map)  
    
nyc_map
    

