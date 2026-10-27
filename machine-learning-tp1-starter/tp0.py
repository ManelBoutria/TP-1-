import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Chargement
D = pd.read_csv("Iris_Dataset.csv")

# 2. Inspection
D.info()
print(D.describe())
print(D.shape)
print(D.columns)
print(D.head())

# 3. Sélection de deux features
X2 = D[["sepal_length", "petal_width"]]

# 4. 20 premières observations
print(X2.head(20))

# 5. Copie du DataFrame original
D_copy = D.copy()

# 6-7. Split 80/20 avec stratification
X = D.drop(columns=["species"])   # features
y = D["species"]                  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Vérification
print(len(X_train), len(X_test))
print(len(X_train) / len(D), len(X_test) / len(D))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))