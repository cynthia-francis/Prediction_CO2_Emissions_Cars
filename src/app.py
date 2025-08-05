import joblib
import streamlit as st
import pandas as pd

# Load the trained model pipeline
model = joblib.load('src/co2_emission_predictor.pkl')

# 2014 Bonus/Malus function (example thresholds, adapt as needed)
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

# Streamlit UI
st.title("Car CO₂ Emissions Predictor & 2014 Bonus/Malus Calculator")

# Input form
power = st.number_input('Power (puiss_admin_98)', min_value=20, max_value=400, value=100)
fuel_type = st.selectbox('Fuel Type (cod_cbr)', options=['FE', 'GO', 'GP/ES', 'ES', 'GN'])
gearbox = st.selectbox('Gearbox type (typ_boite_nb_rapp)', options=[4,5,6,7])

# Button to predict
if st.button("Predict CO₂ & Bonus/Malus"):
    # Prepare input dataframe with required features
    input_dict = {
        'puiss_admin_98': [power],
        'cod_cbr': [fuel_type],
        'typ_boite_nb_rapp': [gearbox],
        'lib_mrq': ['default'],
        'lib_mod_doss': ['default'],
        'lib_mod': ['default'],
        'dscom': ['default'],
        'hybride': [0],
        'conso_urb': [5],
        'conso_exurb': [4],
        'masse_ordma_min': [1200],
        'champ_v9': ['default'],
        'Carrosserie': ['default'],
        'gamme': ['default']
    }
    input_df = pd.DataFrame(input_dict)

    # Convert categorical columns to string dtype to avoid errors during encoding
    categorical_cols = ['cod_cbr', 'lib_mrq', 'lib_mod_doss', 'lib_mod', 'dscom', 'champ_v9', 'Carrosserie', 'gamme']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    # Predict CO2 emissions
    co2_pred = model.predict(input_df)[0]
    st.write(f"Predicted CO₂ emissions: {co2_pred:.2f} g/km")

    # Calculate Bonus/Malus
    bm = bonus_malus_2014(co2_pred)
    st.write(f"Bonus/Malus status (2014 rules): {bm}")
# Save the app as 'src/app.py'
