def process():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    file_path = r"C:\Users\COMPUMARTS\Downloads\Mall_Customers.csv"
    df = pd.read_csv(file_path)
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled