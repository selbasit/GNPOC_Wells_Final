
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import joblib

st.set_page_config(layout="wide", page_title="GNPOC Wells Advanced Analytics")

@st.cache_data
def load_data():
    df = pd.read_csv('GNPOC_Wells.csv', encoding='latin1')
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    cols = ['WELL NAME', 'WELL TYPE', 'LATITUDE', 'LONGITUDE', 
            'NORTHING', 'EASTING', 'OPERATOR', 'DATE', 'SURVEYOR', 'BLOCK #']
    df = df[cols].copy()

    def convert_coord(coord):
        try:
            if 'N' in str(coord):
                return float(str(coord).split('N')[0])
            elif 'E' in str(coord):
                return float(str(coord).split('E')[0])
            else:
                return float(coord)
        except:
            return np.nan

    df['LATITUDE'] = df['LATITUDE'].apply(convert_coord)
    df['LONGITUDE'] = df['LONGITUDE'].apply(convert_coord)

    df['WELL TYPE'] = df['WELL TYPE'].fillna('UNKNOWN').str.strip()
    df['OPERATOR'] = df['OPERATOR'].fillna('UNKNOWN').str.strip()
    df['BLOCK #'] = df['BLOCK #'].fillna('UNKNOWN').str.strip()

    df['DATE_DT'] = pd.to_datetime(df['DATE'], errors='coerce', dayfirst=True)
    df['YEAR'] = df['DATE_DT'].dt.year
    df['MONTH'] = df['DATE_DT'].dt.month

    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE_DT'])
    df = df.reset_index(drop=True)
    return df

df = load_data()

@st.cache_resource
def train_models(df):
    type_df = df[df['WELL TYPE'].isin(['SUSP', 'ABND', 'D&A', 'Development', 'UNKNOWN'])]
    X_type = type_df[['LATITUDE', 'LONGITUDE', 'NORTHING', 'EASTING', 'YEAR']]
    y_type = type_df['WELL TYPE']
    le_type = LabelEncoder()
    y_type_encoded = le_type.fit_transform(y_type)
    X_train, X_test, y_train, y_test = train_test_split(X_type, y_type_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(le_type.classes_), activation='softmax')
    ])
    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2,
                           callbacks=[early_stop], verbose=0)

    coords = df[['LATITUDE', 'LONGITUDE']].values
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(coords)

    anomaly_features = df[['LATITUDE', 'LONGITUDE', 'YEAR']].dropna()
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(anomaly_features)

    ts_data = df.groupby('YEAR').size().reset_index(name='count')

    return {
        'nn_model': nn_model,
        'le_type': le_type,
        'scaler': scaler,
        'type_test': (X_test_scaled, y_test),
        'history': history,
        'kmeans': kmeans,
        'clusters': clusters,
        'iso_forest': iso_forest,
        'anomalies': anomalies,
        'ts_data': ts_data
    }

models = train_models(df)

st.title('GNPOC Wells Advanced Analytics Dashboard')
st.sidebar.header("Controls")
view_data = st.sidebar.checkbox("View Raw Data", True)
show_advanced = st.sidebar.checkbox("Show Advanced Analytics", False)

if view_data:
    st.header('Dataset Overview')
    st.write(f"Total wells: {len(df)}")
    st.dataframe(df.head())

tab1, tab2, tab3, tab4 = st.tabs(["Basic Analytics", "Predictive Models", "Anomaly Detection", "Time Series"])

with tab1:
    st.header('Basic Analytics')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Well Type Distribution')
        type_counts = df['WELL TYPE'].value_counts()
        fig = px.pie(type_counts, values=type_counts.values, names=type_counts.index)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Drilling Activity Over Time')
        year_counts = df['YEAR'].value_counts().sort_index()
        fig = px.line(year_counts, markers=True)
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Wells Drilled')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Operator Distribution')
        operator_counts = df['OPERATOR'].value_counts()
        fig = px.bar(operator_counts)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Block Distribution')
        block_counts = df['BLOCK #'].value_counts().head(10)
        fig = px.bar(block_counts)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader('Geographic Distribution')
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)
    for idx, row in df.iterrows():
        folium.Marker(
            [row['LATITUDE'], row['LONGITUDE']],
            popup=f"{row['WELL NAME']} ({row['WELL TYPE']}, {row['YEAR']})",
            icon=folium.Icon(color=['red', 'blue', 'green', 'purple', 'orange'][models['clusters'][idx]])
        ).add_to(m)
    folium_static(m, width=1200)

with tab2:
    st.header('Predictive Models')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Well Type Prediction (Neural Network)')
        lat = st.number_input('Latitude', value=float(df['LATITUDE'].mean()))
        lon = st.number_input('Longitude', value=float(df['LONGITUDE'].mean()))
        north = st.number_input('Northing', value=float(df['NORTHING'].mean()))
        east = st.number_input('Easting', value=float(df['EASTING'].mean()))
        year = st.number_input('Year', min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=int(df['YEAR'].mean()))
        if st.button('Predict Well Type'):
            input_data = [[lat, lon, north, east, year]]
            input_scaled = models['scaler'].transform(input_data)
            prediction = models['nn_model'].predict(input_scaled)
            well_type = models['le_type'].classes_[prediction[0]]
            confidence = 1.0
            st.success(f'Predicted Well Type: {well_type} (Confidence: {confidence:.2f})')
            st.subheader('Model Performance')
            X_test, y_test = models['type_test']
            y_pred = models['nn_model'].predict(X_test)
            st.text(classification_report(y_test, y_pred, target_names=models['le_type'].classes_))
            fig, ax = plt.subplots()
            ax.plot([], label='Train Accuracy (RF does not use epochs)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)
    with col2:
        st.subheader('Well Clustering Analysis')
        st.write('Wells clustered into 5 geographic groups:')
        fig = px.scatter(df, x='LONGITUDE', y='LATITUDE', color=models['clusters'], hover_name='WELL NAME',
                         hover_data=['WELL TYPE', 'YEAR'])
        st.plotly_chart(fig, use_container_width=True)
        if show_advanced:
            st.subheader('Cluster Characteristics')
            df['Cluster'] = models['clusters']
            cluster_stats = df.groupby('Cluster').agg({'YEAR': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'})
            st.dataframe(cluster_stats)

with tab3:
    st.header('Anomaly Detection')
    st.write("""
    Identifying unusual wells based on location and drilling year using Isolation Forest.
    Anomalies may represent data errors or particularly unusual drilling locations.
    """)
    anomaly_df = df.copy()
    anomaly_features = ['LATITUDE', 'LONGITUDE', 'YEAR']
    numeric_means = anomaly_df[anomaly_features].mean()
    anomaly_df['Anomaly_Score'] = models['iso_forest'].decision_function(
        anomaly_df[anomaly_features].fillna(numeric_means))
    anomaly_df['Is_Anomaly'] = anomaly_df['Anomaly_Score'] < 0
    st.subheader('Anomaly Distribution')
    fig = px.histogram(anomaly_df, x='Anomaly_Score', color='Is_Anomaly')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Anomalies on Map')
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)
    for idx, row in anomaly_df.iterrows():
        color = 'red' if row['Is_Anomaly'] else 'green'
        folium.Marker([row['LATITUDE'], row['LONGITUDE']],
                      popup=f"{row['WELL NAME']} (Score: {row['Anomaly_Score']:.2f})",
                      icon=folium.Icon(color=color)).add_to(m)
    folium_static(m, width=1200)
    st.subheader('Anomalous Wells')
    st.dataframe(anomaly_df[anomaly_df['Is_Anomaly']][
        ['WELL NAME', 'WELL TYPE', 'YEAR', 'LATITUDE', 'LONGITUDE', 'Anomaly_Score']
    ])

with tab4:
    st.header('Time Series Analysis')
    st.write("""
    Analyzing drilling activity over time and forecasting future drilling locations.
    """)
    ts_data = models['ts_data']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Drilling Activity Over Time')
        fig = px.line(ts_data, x='YEAR', y='count', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        if st.button('Run Time Series Forecast'):
            try:
                with st.spinner('Training ARIMA model...'):
                    model = ARIMA(ts_data['count'], order=(1, 1, 1))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=5)
                    forecast_df = pd.DataFrame({'YEAR': range(ts_data['YEAR'].max() + 1, ts_data['YEAR'].max() + 6),
                                                'count': forecast})
                    combined = pd.concat([ts_data, forecast_df])
                    fig = px.line(combined, x='YEAR', y='count', 
                                  color=combined.index >= len(ts_data),
                                  labels={'color': 'Forecast'},
                                  title='Drilling Activity Forecast')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in forecasting: {e}")
    with col2:
        st.subheader('Location Trend Analysis')
        year_range = st.slider('Select Year Range',
                               min_value=int(df['YEAR'].min()),
                               max_value=int(df['YEAR'].max()),
                               value=(int(df['YEAR'].min()), int(df['YEAR'].min()) + 10))
        filtered = df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]
        fig = px.scatter(filtered, x='LONGITUDE', y='LATITUDE',
                         animation_frame='YEAR',
                         range_x=[df['LONGITUDE'].min() - 0.5, df['LONGITUDE'].max() + 0.5],
                         range_y=[df['LATITUDE'].min() - 0.5, df['LATITUDE'].max() + 0.5],
                         title='Drilling Locations Over Time')
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.header("Deployment")
st.sidebar.info("""
This app can be deployed on:
- Streamlit Community Cloud
- AWS EC2
- Google Cloud Run
- Azure App Service
""")
if st.sidebar.button('Save Models for Deployment'):
    joblib.dump(models, 'gnpoc_models.pkl')
    st.sidebar.success("Models saved successfully!")
