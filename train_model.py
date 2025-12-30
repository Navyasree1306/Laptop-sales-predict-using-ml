import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pickle


data = [
    {"brand": "Dell", "cpu": "i5", "ram": 8, "storage": "SSD", "gpu": "Integrated", "age_months": 4, "price_usd": 799.99, "sold": 1},
    {"brand": "HP", "cpu": "i7", "ram": 16, "storage": "SSD", "gpu": "Dedicated", "age_months": 12, "price_usd": 1200.00, "sold": 0},
    {"brand": "Apple", "cpu": "M1", "ram": 8, "storage": "SSD", "gpu": "Integrated", "age_months": 6, "price_usd": 999.00, "sold": 1},
    {"brand": "Asus", "cpu": "Ryzen 5", "ram": 16, "storage": "HDD", "gpu": "Dedicated", "age_months": 24, "price_usd": 850.00, "sold": 0},
    {"brand": "Lenovo", "cpu": "i3", "ram": 4, "storage": "SSD", "gpu": "Integrated", "age_months": 2, "price_usd": 400.00, "sold": 1},
]

df = pd.DataFrame(data)
X = df.drop("sold", axis=1)
y = df["sold"]

# 2. Define Preprocessing
categorical_features = ["brand", "cpu", "storage", "gpu"]
numerical_features = ["ram", "age_months", "price_usd"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# 3. Create Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier())
])

# 4. Train Model
print("Training model...")
model.fit(X, y)

# 5. Save Model
output_file = "laptop_sales_model.pkl"
with open(output_file, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {output_file}")
# 6. Confirm saving
with open(output_file, "rb") as f:
    loaded_model = pickle.load(f)   
print("Model loaded successfully for confirmation.")
print("Training complete.")

