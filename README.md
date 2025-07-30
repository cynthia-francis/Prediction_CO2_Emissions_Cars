# 🚗 CO₂ Emissions Prediction & Bonus-Malus Mapping (France, 2014)

This project uses official 2014 vehicle homologation data from ADEME and UTAC to train machine learning models that predict the CO₂ emissions of cars based on their technical specifications. It also connects these predictions to the official French **bonus-malus écologique** policy from 2014, quantifying the financial incentives or penalties that would apply based on emissions levels.

---

## 📊 Dataset

The data was provided by UTAC (Union Technique de l’Automobile, du motocycle et du Cycle) to ADEME and includes:

- Fuel consumption  
- CO₂ emissions (g/km)  
- Regulated air pollutants  
- Vehicle characteristics (make, model, energy type, CNIT, etc.)

---

## 🧠 Objective

1. **Train ML models** to predict CO₂ emissions from car specs.
2. **Evaluate model performance** using standard metrics (MAE, RMSE, R²).
3. **Map predicted emissions** to bonus-malus values using the 2014 official French grid.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy, Matplotlib/Seaborn  
- Scikit-learn  
- (Optional) XGBoost / LightGBM

---

## 🗺️ Bonus-Malus Logic (France, 2014)

The 2014 scheme awarded a **bonus** to low-emission vehicles and imposed a **malus** (penalty) on high-emission ones. Here are a few examples:

| CO₂ Emissions (g/km) | Bonus/Malus (€) |
|----------------------|------------------|
| ≤ 20                 | +6 300           |
| 21–50                | +4 000           |
| 131–135              | –150             |
| 191–200              | –6 500           |
| > 200                | –8 000           |

See full mapping logic in [`bonus_malus_2014.py`](./bonus_malus_2014.py) (or integrate directly in the notebook).

---

## 📂 Project Structure

