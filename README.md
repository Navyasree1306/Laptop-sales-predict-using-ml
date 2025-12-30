# Machine Learning Project — Laptop Sales Classifier
""
Laptop Sales Prediction using Machine Learning

This project builds a machine learning classification model to predict whether a laptop will be sold based on its specifications such as brand, CPU, RAM, storage, GPU, age, and price.

The model uses a Decision Tree Classifier along with preprocessing handled through a Scikit-learn pipeline.
This project contains a trained RandomForest model that predicts whether a laptop will be sold within 30 days based on its specifications'''

## Files
- `laptop_sales_model.pkl` — The trained ML model.

## How to Use
```python
import pickle, pandas as pd

model = pickle.load(open("laptop_sales_model.pkl","rb"))

sample = pd.DataFrame([{
    "brand":"Dell",
    "cpu":"i5",
    "ram":8,
    "storage":"SSD",
    "gpu":"Integrated",
    "age_months":4,
    "price_usd":799.99
}])

print(model.predict(sample))
```

## Project Details

This classifier analyzes various laptop specifications to predict sales likelihood. The model was developed using a synthetic dataset and can be adapted for real-world applications with actual sales data.

### Model Performance

The RandomForest classifier evaluates features like processor type, memory capacity, and market price to make predictions. For best results, retrain the model periodically as market conditions and hardware specifications evolve.

### Future Improvements

- Integrate real sales data for improved accuracy
- Add feature importance analysis
- Implement cross-validation metrics
- Monitor model drift in production environments

