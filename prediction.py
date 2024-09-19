import joblib

def predict(data):
    model = joblib.load('random_forest_model.sav')
    return model.predict(data)
