import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ======= Define feature lists EXACTLY as in training =========

features = [
    'cod_cbr', 'hybride',
    'puiss_admin_98', 'conso_urb', 'conso_exurb',
    'gearbox_type', 'num_gears',
    'masse_ordma_avg', 'Carrosserie', 'gamme'
]

numeric_features = ['puiss_admin_98', 'conso_urb', 'conso_exurb', 'masse_ordma_avg', 'num_gears']
categorical_features = list(set(features) - set(numeric_features))

# ======= Load Preprocessor & Models ==========

preprocessor = joblib.load('preprocessor.pkl')
models = {}

for model_name in ['Lasso', 'Ridge', 'RandomForest', 'XGBoost']:
    if model_name == 'XGBoost':
        booster = xgb.Booster()
        booster.load_model(f'xgb_model_{model_name}.json')
        xgb_model = xgb.XGBRegressor()
        xgb_model._Booster = booster
        models[model_name] = xgb_model
    else:
        models[model_name] = joblib.load(f'model_{model_name}.pkl')

# ======= Define a prediction helper to cast types correctly ==========

def prepare_input(df):
    """Ensure correct dtypes before transforming."""
    # Cast numerics
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Cast categoricals
    for col in categorical_features:
        df[col] = df[col].astype(str)
    # Ensure column order and presence
    df = df[all_features]
    return df

def make_pipeline(preprocessor, model):
    # Returns a function that preprocesses and predicts given X
    def predict(X):
        # Prepare X
        X_clean = prepare_input(X.copy())
        # Transform and predict
        X_prep = preprocessor.transform(X_clean)
        return model.predict(X_prep)
    return predict

predictors = {name: make_pipeline(preprocessor, mdl) for name, mdl in models.items()}

# ======= Bonus/Malus Calculator ==========

def bonus_malus_2014(co2):
    if co2 <= 100:
        return "Bonus: 1000€"
    elif co2 <= 130:
        return "No bonus or malus"
    elif co2 <= 160:
        return "Malus: 200€"
    elif co2 <= 190:
        return "Malus: 1000€"
    else:
        return "Malus: 2000€"

# ======= Streamlit UI ==========

st.title("Car CO₂ Emissions Predictor & 2014 Bonus/Malus Calculator")
selected_model = st.selectbox("Choose prediction model", options=list(models.keys()))

# Input widgets for user input (adjusted and complete)

# --- Numeric inputs ---
power = st.number_input('Power (puiss_admin_98)', min_value=20, max_value=400, value=100)
conso_urb = st.number_input('Urban Consumption (conso_urb)', min_value=0.0, max_value=50.0, value=5.0)
conso_exurb = st.number_input('Extra-urban Consumption (conso_exurb)', min_value=0.0, max_value=50.0, value=4.0)
masse_ordma_avg = st.number_input('Mass (masse_ordma_avg)', min_value=500.0, max_value=3000.0, value=1200.0)
num_gears = st.selectbox('Number of Gears (num_gears)', options=[0, 4, 5, 6, 7, 8, 9])

# --- Categorical inputs ---
fuel_type = st.selectbox('Fuel Type (cod_cbr)', options=['ES', 'GO', 'ES/GP', 'GP/ES', 'EH', 'GH', 'ES/GN', 'GN/ES', 'FE', 'GN', 'GL'])
hybride = st.selectbox('Is it a hybrid vehicle? (hybride)', options=['non', 'oui'])
gearbox_type = st.selectbox('Gearbox Type (gearbox_type)', options=['M', 'A', 'D', 'V', 'S'])
carrosserie = st.selectbox('Body Type (Carrosserie)', options=[
    'BERLINE', 'BREAK', 'COUPE', 'CABRIOLET', 'TS TERRAINS/CHEMINS',
    'COMBISPACE', 'MINISPACE', 'MONOSPACE COMPACT', 'MONOSPACE', 'MINIBUS', 'COMBISPCACE'
])
gamme = st.selectbox('Gamme', options=['MOY-SUPER', 'LUXE', 'MOY-INFER', 'INFERIEURE', 'SUPERIEURE', 'ECONOMIQUE'])

# --- Build input dictionary ---
input_dict = {
    'puiss_admin_98': [power],
    'conso_urb': [conso_urb],
    'conso_exurb': [conso_exurb],
    'masse_ordma_avg': [masse_ordma_avg],
    'num_gears': [num_gears],
    'cod_cbr': [fuel_type],
    'hybride': [hybride],
    'gearbox_type': [gearbox_type],
    'Carrosserie': [carrosserie],
    'gamme': [gamme]
}
# Convert input dictionary to DataFrame
input_df = pd.DataFrame(input_dict)
input_df = input_df[features]  # Ensure column order

# Prediction button
if st.button("Predict CO₂ & Bonus/Malus"):
    try:
        # See the data passed for sanity check (optional debug)
        # st.write("Prepared input:", input_df)
        pred_func = predictors[selected_model]
        co2_pred = pred_func(input_df)[0]
        st.success(f"Predicted CO₂ emissions by {selected_model}: {co2_pred:.2f} g/km")
        bm = bonus_malus_2014(co2_pred)
        st.info(f"Bonus/Malus status (2014 rules): {bm}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Input DataFrame:", input_df)
