import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Simulate realistic data
data = {
    "start": ["N1", "N2", "N3", "N4", "N5"],
    "end": ["R1", "R1", "R1", "R1", "R1"],
    "time_of_day": [8, 10, 14, 16, 20],
    "traffic_level": [3, 6, 5, 7, 2],
    "weather": ["Clear", "Rain", "Fog", "Clear", "Rain"],
    "delay": [6, 10, 12, 7, 9]
}
df = pd.DataFrame(data)
df.to_csv("data/traffic_data.csv", index=False)

# ML model
df['weather'] = LabelEncoder().fit_transform(df['weather'])
X = df[['time_of_day', 'traffic_level', 'weather']]
y = df['delay']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'models/delay_model.pkl')
joblib.dump(LabelEncoder().fit(data['weather']), 'models/weather_encoder.pkl')
