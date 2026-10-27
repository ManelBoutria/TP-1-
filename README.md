# Machine Learning - Practical Session 1

Python for Data Exploration and Preprocessing

Files: `tp1.py` (code), `Iris_Dataset.csv`, `txt_file.txt`, `employees.csv`.

---

## Exercise 1 - Exploring a tabular dataset and creating data splits

**2. What each output tells you**
1. `info()`: number of rows, column names, data types and non-null counts (shows missing values).
2. `describe()`: mean, standard deviation, min, max and quartiles of numeric columns (scales, spread, possible outliers).
3. `shape`: (rows, columns). For Iris: (150, 5).
4. `columns`: the list of column names.
5. `head()`: the first 5 rows, to check the file was loaded correctly.

**5. Why a copy?** A copy lets us experiment with preprocessing (scaling, imputation, dropping columns) without altering the original data. If a transformation goes wrong, we go back to the original without reloading the file. Without `.copy()`, `B = A` only creates a second name for the same object.

**7. Variables**
- Input features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
- Target: `species`.
- `stratify=y` keeps the same class proportions in the training and test sets.

**8. Verification (150 observations)**
- Training set: 120 (80 %), 40 per class.
- Test set: 30 (20 %), 10 per class.

**Discussion: why keep the test set separate until the final evaluation?**
The test set must simulate data the model has never seen. If its information influences preprocessing (for example computing the mean and standard deviation for scaling on all the data) or model selection (tuning hyperparameters by looking at the test score), we get **data leakage**. Consequences:
- The measured performance is too optimistic.
- The model has indirectly adapted to the test set, so it overfits it.
- Real performance in production is lower than reported.

Good practice: fit preprocessing on the training set only, then apply it to the test set; select models with cross-validation on the training set; use the test set once, at the end.

---

## Exercise 2 - Exploring text data

**4. Dimensions**
- `len(news.data)` = number of documents: 11,314 in the training subset.
- `news.target.shape` = `(11314,)`: one numeric label per document.

**6. Target values and class names**
`target_names` is a list of 20 class names. Each number in `target` is the index of the class in that list. For example, `target_names[target[0]]` gives the class name of the first document.

---

## Exercise 3 - Detecting and handling missing values

**5. Why are `'-'` and `'na'` not detected?**
For pandas, only `None`, `np.nan` and `pd.NA` are missing values. `'-'` and `'na'` are ordinary strings, and pandas cannot know that they mean "missing" in this dataset.

**6. Conversion rule**
The strings `'-'`, `'na'`, `'n/a'` and empty strings are converted to `np.nan` after removing whitespace and lowercasing, so that `'NA'`, `' na '` and `'na'` are treated the same way.

**7. Imputation choices**
- Categorical columns (`Gender`, `Senior Management`, `Team`): `most_frequent`.
- Numerical columns (`Salary`, `Bonus %`): `median` (robust to outliers).
- `First Name` (identifier) and the date columns are not imputed: inventing a name or a date has no justification.
- The imputers are fitted on the training set only and then applied to the test set, to avoid data leakage.

**8. Limitations of imputation**
- It invents values, so the data looks more complete than it is.
- It reduces the variance of the imputed columns.
- It can bias results if the values are not missing at random.
- A single value (median or mode) ignores relations between columns.

---

## Exercise 4 - Feature scaling, binarization and encoding

### Part A

**3. Main idea of each method**

| Method | Idea | Acts on |
|---|---|---|
| StandardScaler | (x - mean) / std: mean 0, std 1 | each feature |
| MinMaxScaler | rescales to the range [0, 1] | each feature |
| MaxAbsScaler | divides by the maximum absolute value: range [-1, 1], zeros are kept | each feature |
| RobustScaler | (x - median) / IQR: resistant to outliers | each feature |
| normalize | rescales each row to unit norm (L2 by default) | each sample |

**4. Binarization**
Values strictly above the threshold become 1, the others become 0. With threshold 0, the positive values give 1. With threshold 1, only values above 1 give 1, so there are fewer ones. A higher threshold produces fewer ones.

**Discussion: why does scaling matter?**
- KNN and SVM rely on distances: a feature with a large scale dominates the others.
- Gradient-based models converge faster and more stably when features have similar scales.
- Scaling is not necessary for decision trees: a split only depends on the order of values within one feature, not on their scale.

### Part B

**1. OrdinalEncoder**
It assigns an integer code to each category of each column (alphabetical order by default, for example female = 0, male = 1). These numbers are category codes, not rankings: they do not mean that male is greater than female.

**3. Integer label encoding vs one-hot encoding**
- Integer encoding: one column with arbitrary numbers, so a model may read a false order or distance between classes.
- One-hot encoding: one binary column per category, with a single 1 per row, so no order is implied. The cost is more columns.

**4. Why one-hot for nominal input features?**
Nominal features (country, browser, gender) have no natural order. Arbitrary ordinal codes introduce a fake order and fake distances (for example Africa < Asia < Europe), which can mislead distance-based and linear models. One-hot encoding avoids this.

---

## Conclusion

| Data type | Appropriate preprocessing |
|---|---|
| Numerical | impute with the median, then scale (Standard, MinMax or Robust) |
| Categorical | impute with the most frequent value, then one-hot encode (ordinal encoding only if there is a true order) |
| Text | clean, tokenize, then vectorize (for example bag of words or TF-IDF) |

Every preprocessing step must be fitted on the training set only and then applied to the test set.