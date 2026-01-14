# AeroRoute - Airport Route Optimizer

AeroRoute is a sophisticated tool designed to optimize ground movement at New Delhi Airport. It leverages graph algorithms and machine learning to predict the most efficient taxiing routes, accounting for real-time factors like traffic and weather.

## Features
- **Shortest Path Calculation**: Uses Dijkstra's algorithm to find the optimal path.
- **Delay Prediction**: Machine Learning model (Random Forest) predicts operational delays.
- **Interactive Visualization**: Visualizes the airport node network and calculated paths.
- **Real-time Configuration**: Adjust parameters like time of day, traffic level, and weather.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Data & Models**:
   ```bash
   python utils/label_nodes_delhi.py
   python models/train_model.py
   ```

3. **Run Application**:
   ```bash
   python app.py
   ```
   Access at `http://127.0.0.1:5000`.

## Tech Stack
- **Backend**: Python, Flask, NetworkX, Pandas, Scikit-Learn
- **Frontend**: HTML5, Bootstrap 5, Vis.js

## Tech Stack
<img width="1470" height="830" alt="AeroRoute_P1" src="https://github.com/user-attachments/assets/d4b87ccb-5d3f-4ea7-933e-4c27b82f5d6a" />
