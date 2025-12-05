import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import pickle
import os
from phik import phik_matrix

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Price Prediction & Analysis")

def prepare_data(df):
    def clean_num(x):
        if pd.isna(x): return np.nan
        x = str(x).lower()
        x = x.replace('kmpl', '').replace('km/kg', '').replace('cc', '').replace('bhp', '').strip()
        if not x: return np.nan
        try: return float(x)
        except ValueError: return np.nan

    def parse_torque(t):
        if pd.isna(t): return pd.Series([np.nan, np.nan])
        t = str(t).lower()
        t = re.sub(r'[ ,]', '', t)
        t = t.replace('(kgm@rpm)', '').replace('kgm@', 'kgm').replace('nm@', 'nm')
        torque_match = re.search(r'([\d\.]+)(nm|kgm)?', t)
        torque = float(torque_match.group(1)) if torque_match else np.nan
        if torque_match and torque_match.group(2) == 'kgm': torque *= 9.80665
        rpm_match = re.search(r'(\d+)-(\d+)rpm|@(\d+)rpm|(\d+)rpm', t)
        if rpm_match:
            nums = [x for x in rpm_match.groups() if x is not None]
            rpm = int(nums[0])
        else: rpm = np.nan
        return pd.Series([torque, rpm])

    def preprocess(df):
        df = df.copy()
        df['mileage'] = df['mileage'].apply(clean_num)
        df['engine'] = df['engine'].apply(clean_num)
        df['max_power'] = df['max_power'].apply(clean_num)
        df[['torque', 'max_torque_rpm']] = df['torque'].apply(parse_torque)
        cols_to_num = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
        for col in cols_to_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    df = preprocess(df)

    if 'engine' in df.columns and 'seats' in df.columns:
        df[['engine', 'seats']] = df[['engine', 'seats']].fillna(0).astype(int)

    if 'name' in df.columns:
        df["brand"] = df["name"].str.split(" ", n=1).str[0]

    return df

@st.cache_data
def load_dataset():
    df = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df = prepare_data(df)
    df = df.drop_duplicates(subset=df.columns.drop('selling_price'), keep='first').reset_index(drop=True)
    for c in ['mileage', 'engine', 'max_power', 'seats', 'year']:
        df[c] = df[c].fillna(df[c].median())
    return df

df_full = load_dataset()

page = st.sidebar.radio("Mode", ["EDA", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.header("Filter Settings")
brands = st.sidebar.multiselect("Brands", sorted(df_full['brand'].unique()), default=df_full['brand'].value_counts().index[:5])
min_year = st.sidebar.slider("Year from", int(df_full['year'].min()), int(df_full['year'].max()), 2010)

if brands:
    df_viz = df_full[(df_full['brand'].isin(brands)) & (df_full['year'] >= min_year)]
else:
    df_viz = df_full[df_full['year'] >= min_year]

if page == "EDA":
    st.header("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Numerical", "Categorical", "Correlation", "Price Analysis"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            col = st.selectbox("Feature Distribution", ['km_driven', 'year', 'mileage', 'engine', 'max_power'])
            st.plotly_chart(px.histogram(df_viz, x=col, color="owner", marginal="box", title=f"Distribution of {col}"), use_container_width=True)
        with c2:
            st.plotly_chart(px.box(df_viz, y=col, x="transmission", title=f"{col} by Transmission"), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            cat_col = st.selectbox("Categorical Feature", ['fuel', 'seller_type', 'transmission', 'owner'])
            counts = df_viz[cat_col].value_counts().reset_index()
            counts.columns = [cat_col, 'count']
            st.plotly_chart(px.bar(counts, x=cat_col, y='count', color=cat_col, title=f"Count of {cat_col}"), use_container_width=True)
        with c2:
            st.plotly_chart(px.box(df_viz, x=cat_col, y="selling_price", title=f"Price by {cat_col}"), use_container_width=True)

    with tab3:
        st.subheader("Correlation Matrix")
        corr_type = st.radio("Method", ["Pearson", "Phik"], horizontal=True)

        if corr_type == "Pearson":
            corr = df_viz.select_dtypes(include=np.number).corr()
            st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
        else:
            if len(df_viz) > 500:
                st.warning("Subsampling to 500 rows for performance.")
                df_phik = df_viz.sample(500)
            else:
                df_phik = df_viz

            cols_phik = ['selling_price', 'year', 'km_driven', 'fuel', 'transmission', 'max_power', 'brand']
            st.plotly_chart(px.imshow(phik_matrix(df_phik[cols_phik]), text_auto=".2f", color_continuous_scale="Blues"), use_container_width=True)

    with tab4:
        x_ax = st.selectbox("X Axis", ['max_power', 'engine', 'year', 'km_driven'])
        st.plotly_chart(px.scatter(df_viz, x=x_ax, y="selling_price", color="transmission", log_y=True, title="Price Dependency (Log Scale)"), use_container_width=True)

elif page == "Prediction":
    st.header("Price Prediction Model")

    @st.cache_resource
    def load_ml_bundle():
        path = 'models/final_pipeline.pkl'
        if not os.path.exists(path):
            st.error("Model file not found.")
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    artifacts = load_ml_bundle()

    def predict_dataframe(df_input, artifacts):
        df_clean = prepare_data(df_input)

        medians = artifacts['medians']
        for col, val in medians.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(val)

        if 'km_driven' in df_clean.columns:
            df_clean['km_driven_log'] = np.log1p(df_clean['km_driven'].astype(float))

        try:
            X = artifacts['preprocessor'].transform(df_clean)
            log_preds = artifacts['model'].predict(X)
            return np.expm1(log_preds)
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            return None

    if artifacts:
        ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Single Input", "CSV Upload", "Model Weights"])

        with ml_tab1:
            with st.form("calc_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    name = st.text_input("Car Name", "Maruti Swift Dzire VDI")
                    year = st.number_input("Year", 1990, 2025, 2015)
                    km = st.number_input("Kilometers Driven", 0, 1000000, 50000)
                    fuel = st.selectbox("Fuel", ['Diesel', 'Petrol', 'CNG', 'LPG'])
                with c2:
                    seller = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
                    trans = st.selectbox("Transmission", ['Manual', 'Automatic'])
                    owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner'])
                    mileage = st.text_input("Mileage", "23.4 kmpl")
                with c3:
                    engine = st.text_input("Engine", "1248 CC")
                    power = st.text_input("Max Power", "74 bhp")
                    seats = st.number_input("Seats", 2, 10, 5)

                submit = st.form_submit_button("Calculate Price")

            if submit:
                row = pd.DataFrame([{
                    'name': name, 'year': year, 'km_driven': km, 'fuel': fuel,
                    'seller_type': seller, 'transmission': trans, 'owner': owner,
                    'mileage': mileage, 'engine': engine, 'max_power': power, 'seats': seats, 'torque': np.nan
                }])

                price = predict_dataframe(row, artifacts)
                if price is not None:
                    st.success(f"Predicted Price: **{price[0]:,.0f}** ₹")

        with ml_tab2:
            st.subheader("Batch Prerdiction")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")

            if uploaded_file:
                df_uploaded = pd.read_csv(uploaded_file)
                st.write("Preview:")
                st.dataframe(df_uploaded.head())

                if st.button("Predict All"):
                    with st.spinner("Processing..."):
                        prices = predict_dataframe(df_uploaded, artifacts)

                        if prices is not None:
                            df_uploaded['Predicted_Price'] = prices
                            st.success("Done!")
                            st.write(df_uploaded[['name', 'Predicted_Price']].head())

                            csv_data = df_uploaded.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv_data,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )

        with ml_tab3:
            st.subheader("Feature Importance")
            try:
                model = artifacts['model']
                preprocessor = artifacts['preprocessor']

                cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
                num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
                all_features = np.r_[cat_names, num_names]

                if len(model.coef_) == len(all_features):
                    imp_df = pd.DataFrame({'Feature': all_features, 'Weight': model.coef_})
                    imp_df['Abs'] = imp_df['Weight'].abs()
                    top_features = imp_df.sort_values('Abs', ascending=False).head(15)

                    fig = px.bar(top_features, x='Weight', y='Feature', orientation='h',
                                 title="Top 15 Features", color='Weight', color_continuous_scale='RdBu')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature mapping mismatch.")
            except Exception as e:
                st.warning(f"Visualization error: {e}")
