import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, normalize, Binarizer,
                                   OrdinalEncoder, LabelEncoder, OneHotEncoder)

print("===== EXERCISE 1 =====")

print("1")
D = pd.read_csv("Iris_Dataset.csv")

print("2")
D.info()
print(D.describe())
print(D.shape)
print(D.columns)
print(D.head())

print("3")
X2 = D[["sepal_length", "petal_width"]]

print("4")
print(X2.head(20))

print("5")
D_copy = D.copy()

print("6-7")
X = D.drop(columns=["species"])
y = D["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("8")
print(len(X_train), len(X_test))
print(len(X_train) / len(D), len(X_test) / len(D))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

print("===== EXERCISE 2 =====")

print("1-2")
with open("txt_file.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.rstrip("\n"))

print("3")
news = fetch_20newsgroups(subset="train")

print("4")
print(len(news.data))
print(news.target.shape)

print("5")
print(news.data[0])
print(news.target[0])

print("6")
print(news.target_names)
print(news.target_names[news.target[0]])

print("7")
lengths = [len(d) for d in news.data]
print(np.mean(lengths), np.max(lengths))
for w in ["the", "god", "car"]:
    print(w, sum(d.lower().split().count(w) for d in news.data))

print("===== EXERCISE 3 =====")

print("1")
E = pd.read_csv("employees.csv")
E.info()
print(E.head())

print("2")
print(E.isna().sum())

print("3")
plt.figure(figsize=(10, 5))
sns.heatmap(E.isna(), cbar=False)
plt.title("Missing values in employees.csv")
plt.xlabel("Columns")
plt.ylabel("Row index")
plt.show()

print("4")
E2 = E.copy()
new_row = pd.DataFrame([{
    "First Name": "-", "Gender": "na",
    "Start Date": np.nan, "Last Login Time": np.nan,
    "Salary": np.nan, "Bonus %": np.nan,
    "Senior Management": np.nan, "Team": np.nan
}])
E2 = pd.concat([E2, new_row], ignore_index=True)

print("5")
print(E2.isna().sum())

print("6")
markers = {"-", "na", "n/a", ""}
for col in E2.select_dtypes(include="object").columns:
    mask = E2[col].astype(str).str.strip().str.lower().isin(markers)
    E2.loc[mask, col] = np.nan
print(E2.isna().sum())

print("7")
cat_cols = ["Gender", "Senior Management", "Team"]
num_cols = ["Salary", "Bonus %"]
E2["Senior Management"] = E2["Senior Management"].astype("object")

train, test = train_test_split(E2, test_size=0.2, random_state=42)
train, test = train.copy(), test.copy()

imp_cat = SimpleImputer(strategy="most_frequent")
imp_num = SimpleImputer(strategy="median")

train[cat_cols] = imp_cat.fit_transform(train[cat_cols])
test[cat_cols] = imp_cat.transform(test[cat_cols])
train[num_cols] = imp_num.fit_transform(train[num_cols])
test[num_cols] = imp_num.transform(test[num_cols])

print("8")
comparison = pd.DataFrame({
    "before (train)": E2.loc[train.index].isna().sum(),
    "after (train)": train.isna().sum(),
    "before (test)": E2.loc[test.index].isna().sum(),
    "after (test)": test.isna().sum(),
})
print(comparison)

print("===== EXERCISE 4 - PART A =====")

M = np.array([[1, -1, 2],
              [2,  0, 0],
              [0,  1, -1]], dtype=float)

print("1-2")
results = {
    "StandardScaler": StandardScaler().fit_transform(M),
    "MinMaxScaler": MinMaxScaler().fit_transform(M),
    "MaxAbsScaler": MaxAbsScaler().fit_transform(M),
    "RobustScaler": RobustScaler().fit_transform(M),
    "normalize (L2)": normalize(M, norm="l2"),
}
for name, R in results.items():
    print(name)
    print(np.round(R, 3))

print("4")
print(Binarizer(threshold=0.0).fit_transform(M))
print(Binarizer(threshold=1.0).fit_transform(M))

print("===== EXERCISE 4 - PART B =====")

data = pd.DataFrame({
    "Feature 1": ["male", "female", "female", "male"],
    "Feature 2": ["from US", "from Europe", "from Asia", "from Africa"],
    "Feature 3": ["uses Safari", "uses Safari", "uses Firefox", "uses Chrome"],
    "Target": ["US", "France", "Japan", "Algeria"],
})
Xc = data[["Feature 1", "Feature 2", "Feature 3"]]
yc = data["Target"]

print("1")
oe = OrdinalEncoder()
print(oe.fit_transform(Xc))
print(oe.categories_)

print("2")
le = LabelEncoder()
le.fit_transform(yc)
print(dict(zip(le.classes_, le.transform(le.classes_))))

print("3")
ohe = OneHotEncoder(sparse_output=False)
print(ohe.fit_transform(yc.to_frame()))
print(ohe.categories_)