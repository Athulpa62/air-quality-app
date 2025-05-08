import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data and model with error handling
try:
    df = pd.read_csv('data/merged_data.csv')
    model = joblib.load('model_rf.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"❌ Failed to load data or model: {e}")
    st.stop()

# Ensure station column exists
if 'station' not in df.columns:
    st.error("🚨 'station' column not found in dataset!")
    st.stop()

# Set page config
st.set_page_config(page_title="Air Quality Forecasting 🌿", page_icon="🌍", layout="wide")

# Station list
stations = ['Dongsi', 'Changping', 'Huairou', 'Aotizhongxin']

# Sidebar
st.sidebar.title("Station Selection 🌍")
selected_station = st.sidebar.selectbox("Choose a Station", stations)

# Filter data
filtered_df = df[df['station'] == selected_station]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Data Overview", "📈 EDA", "⚙️ Predict PM2.5"])

# --- 🏠 Home Tab ---
with tab1:
    st.title(f"🌿 Air Quality Forecasting - {selected_station}")
 # Add a general image

    st.image("air_quality_banner.jpg", use_column_width=True, caption="Clean Air, Healthy Living")

    st.markdown(f"""
    Welcome to the **Air Quality Prediction Platform** for the **{selected_station}** monitoring station!  
    This platform helps you understand, explore, and forecast air pollution using real-time and historical data.

    ### 🧭 What You Can Do Here:
    - 📊 **Explore data** collected from the {selected_station} air quality monitoring station.
    - 📈 **Visualize patterns** and correlations among pollutants and weather conditions.
    - ⚙️ **Predict PM2.5** concentrations using a machine learning model trained on past data.

    ### 🌫️ About PM2.5:
    PM2.5 refers to atmospheric particulate matter (PM) with a diameter of less than 2.5 micrometers.
    These fine particles pose serious health risks because they can penetrate deep into the lungs and even enter the bloodstream.

    ### 🌍 Why Forecast Air Quality?
    Forecasting air quality:
    - Helps citizens take preventive measures to protect their health.
    - Assists city planners and environmental agencies in pollution control.
    - Supports data-driven decision-making for a cleaner future.

    ### 📌 Station Description:
    """)

    station_descriptions = {
        "Dongsi": "📍 Dongsi is an urban site in central Beijing, often used as a reference for city pollution levels.",
        "Changping": "🏞️ Changping is a suburban area northwest of Beijing, capturing regional background pollution.",
        "Huairou": "🌄 Huairou is a rural station with natural surroundings, typically reflecting lower pollution.",
        "Aotizhongxin": "🏟️ Aotizhongxin is near Beijing’s Olympic Green, representing dense urban activity."
    }

    st.info(station_descriptions.get(selected_station, "ℹ️ No description available for this station."))

    st.markdown("Use the tabs above to begin your air quality journey! 🌟")

# --- 📊 Data Overview ---
with tab2:
    st.title(f"📊 Data Overview - {selected_station}")
    st.image("data_overview.jpg", use_column_width=True, caption="Overview of Air Quality Data")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{filtered_df.shape[0]}")
    col2.metric("Columns", f"{filtered_df.shape[1]}")
    col3.metric("Missing %", f"{round(filtered_df.isnull().mean().mean() * 100, 2)}%")

    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.subheader("Missing Values")
    missing = filtered_df.isnull().sum()
    if missing.sum() > 0:
        st.dataframe(missing[missing > 0], use_container_width=True)
    else:
        st.success("✅ No missing values detected.")

# --- 📈 EDA Tab ---
with tab3:
    st.title(f"📈 Exploratory Data Analysis (EDA) - {selected_station}")
    st.image("data_overview.jpg", use_column_width=True, caption="Visualize Patterns in Air Quality")
    st.markdown("---")

    st.subheader("Visualization Options")
    chart_type = st.selectbox("Choose a visualization", ["PM2.5 Distribution", "Correlation Heatmap", "Pairplot"])

    if chart_type == "PM2.5 Distribution":
        st.subheader(f"Distribution of PM2.5 - {selected_station}")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['PM2.5'], bins=50, kde=True, color='skyblue', ax=ax)
        ax.set_xlabel("PM2.5 Concentration (µg/m³)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of PM2.5")
        st.pyplot(fig)

    elif chart_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numeric_df = filtered_df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

    elif chart_type == "Pairplot":
        st.subheader("Pairplot of Selected Features")
        selected_cols = st.multiselect(
            "Select features for pairplot",
            filtered_df.columns.tolist(),
            default=["PM2.5", "PM10", "SO2", "NO2"]
        )
        if selected_cols:
            if len(selected_cols) > 5:
                st.warning("⚠️ Too many features selected. May slow performance.")
            try:
                fig = sns.pairplot(filtered_df[selected_cols])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating pairplot: {e}")

# --- ⚙️ Prediction Tab ---
with tab4:
    st.title(f"⚙️ PM2.5 Prediction - {selected_station}")
    st.image("pm25_health_chart.jpg", use_column_width=True, caption="Forecast PM2.5 Levels")
    st.markdown("---")
    st.write("Adjust the features below to predict PM2.5:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pm10 = st.slider("PM10 (µg/m³)", 0, 1000, 100)
            so2 = st.slider("SO2 (µg/m³)", 0, 500, 15)
            no2 = st.slider("NO2 (µg/m³)", 0, 500, 20)
            co = st.slider("CO (mg/m³)", 0.0, 5.0, 1.0)

        with col2:
            o3 = st.slider("O3 (µg/m³)", 0, 500, 30)
            wspd = st.slider("Wind Speed (m/s)", 0, 20, 5)
            rain = st.slider("Rainfall (mm)", 0, 10, 0)
            temp = st.slider("Temperature (°C)", -20, 40, 15)

        with col3:
            dewp = st.slider("Dew Point (°C)", -20, 40, 5)
            pre = st.slider("Pressure (hPa)", 900, 1100, 1010)
            month = st.slider("Month", 1, 12, 6)
            hour = st.slider("Hour", 0, 23, 12)

        submit = st.form_submit_button("🚀 Predict PM2.5")

    if submit:
        with st.spinner('Predicting... Please wait...'):
            input_data = [[pm10, so2, no2, co, o3, wspd, rain, temp, dewp, pre, month, hour]]
            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                st.success(f"🎯 Predicted PM2.5 Concentration: **{prediction[0]:.2f} µg/m³**")
                st.write("📝 Input Summary:")
                st.dataframe(pd.DataFrame(input_data, columns=[
                    'PM10', 'SO2', 'NO2', 'CO', 'O3', 'WSPM',
                    'RAIN', 'TEMP', 'DEWP', 'PRES', 'month', 'hour'
                ]))
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
