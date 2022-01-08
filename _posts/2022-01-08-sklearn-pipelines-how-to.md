---
title:  "Making Your Data Flow With Sklearn Pipelines"
date:   2022-01-08

description: The basics of Sklearn pipelines.
categories: python sklearn datascience data

excerpt: Sklearn's [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) are an elegant way to organize your modeling workflow.  It also provides an "at-a-glance" picture of what is going into the current model &mdash; something your future self will thank you for when you read that notebook back in six months.
---

## Introduction

Sklearn's [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) are an elegant way to organize your modeling workflow.  It also provides an "at-a-glance" picture of what is going into the current model &mdash; something your future self will thank you for when you read that notebook back in six months.

## Getting Toy Data To Play With

Let's import all the libraries we're working with (don't worry if you don't know what some of these do, we'll get to it!) and get some toy data to work with.  We'll be working with the cute [Penguins](https://github.com/allisonhorst/palmerpenguins) dataset which ``seaborn`` can load.

**Note that I will be emphasizing type hints and style quite a bit!**

**Goal**: We'll try to predict the sex, given the rest of the features.


```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
```


```python
df = sns.load_dataset("penguins")  # type: ignore
```

Great, let's do some quick EDA to see what we're working with.


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>342.000000</td>
      <td>342.000000</td>
      <td>342.000000</td>
      <td>342.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>43.921930</td>
      <td>17.151170</td>
      <td>200.915205</td>
      <td>4201.754386</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.459584</td>
      <td>1.974793</td>
      <td>14.061714</td>
      <td>801.954536</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.100000</td>
      <td>13.100000</td>
      <td>172.000000</td>
      <td>2700.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.225000</td>
      <td>15.600000</td>
      <td>190.000000</td>
      <td>3550.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>44.450000</td>
      <td>17.300000</td>
      <td>197.000000</td>
      <td>4050.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.500000</td>
      <td>18.700000</td>
      <td>213.000000</td>
      <td>4750.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>59.600000</td>
      <td>21.500000</td>
      <td>231.000000</td>
      <td>6300.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().sum(axis=0)
```




    species               0
    island                0
    bill_length_mm        2
    bill_depth_mm         2
    flipper_length_mm     2
    body_mass_g           2
    sex                  11
    dtype: int64



For the missing ``sex`` values, let's drop those rows for now, since we're trying to predict on ``sex``.

If we were to simply do
```python
df.dropna(subset=["sex"], inplace=True)
```
in a cell, we might forget that we applied this and get messed up down the line!  Because Jupyter Notebooks are pretty easy to mess up when you've got code in a bunch of different cells, we're going to make a data import function that does basic importing and cleaning.


```python
def get_and_clean_penguin_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Get and clean ``Penguins`` data.

    Loads ``Penguins`` data, removes rows with null values for ``sex``.
    Returns (df_features, df_target) as a tuple of dataframes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
    """

    df = pd.DataFrame(sns.load_dataset("penguins"))  # type: ignore
    df.dropna(subset=["sex"], inplace=True)

    # Transform Male/Female into 0/1.
    targets: pd.Series = (
        df["sex"].apply(lambda x: 0 if x == "Male" else 1) # type: ignore
    )

    return (df.drop("sex", axis=1), targets)
```

Great, now let's look at our numeric data.  There's a few things to do:

- We'd like to impute on the missing values,
- We'd like to scale these down a bit so everything is nice and normalized.

Let's use a ``Pipeline`` to do this.

A ``Pipeline``will take a list of 2-tuples ``(name, transform)`` where a ``transform`` in Sklean is defined as anything which has implemented the ``fit``/``transform`` methods.


```python
pipeline_numeric = Pipeline([
    ("impute_w_mean", SimpleImputer(strategy="mean")),
    ("scale_normal", StandardScaler())
])
```

We note here that the name of the first step of the pipeline is ``impute_w_mean`` and the associated transform is ``SimpleImputer``.  Similarly, ``scale_normal`` is associated to ``StandardScaler``.

For kicks, let's run this through some data and see what happens.


```python
# Running some fake data through ``pipeline_numeric`` for fun.
fake_data = np.array([1, 2, 2, np.nan, 4, 3, 1, 2, np.nan]).reshape(-1, 1)
pipeline_numeric.fit_transform(fake_data)
```




    array([[-1.30930734],
           [-0.16366342],
           [-0.16366342],
           [ 0.        ],
           [ 2.12762443],
           [ 0.98198051],
           [-1.30930734],
           [-0.16366342],
           [ 0.        ]])



Interesting!  We see here that this replaced our N/A values with whatever the mean was, then normalized our data which sent the mean to 0.  Cool.

---

What about the categorical data?  Can we do anything with that?  Since there are only a few islands (3) and a few species (3), we might try ``OneHotEncoder`` and see what we get from that.  Let's make a similar pipeline, imputing with the most frequent value if necessary.


```python
pipeline_categorical = Pipeline([
    ("impute_w_most_frequent", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encode", OneHotEncoder(handle_unknown='ignore', sparse=False))
])
```

Let's try this one on some fake data as well.


```python
# Running some fake data through ``pipeline_categorical`` for fun.
fake_data = np.array(["a", "a", "b", np.nan, np.nan], dtype=object).reshape(-1, 1)
pipeline_categorical.fit_transform(fake_data)
```




    array([[1., 0.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.]])



Might be a bit harder to tell, but this imputed the missing values as "a", and then converted "a" and "b" to ``[1, 0]`` and ``[0, 1]`` respectively.

---

## Preprocessing: Putting It All Together

So far, we've made our numeric and categorical pipelines for the loaded data. We need to tell Sklearn what pipeline each column should go into.  This is where ``ColumnTransformer`` comes in.  This time, we pass a list of 3-tuples in representing ``(name, pipeline, column names to use)``.

Our preprocessing code, excluding the helper function we made, should look something like this:


```python
# All preprocessing code, excluding helper functions.

pipeline_numeric = Pipeline([
    ("impute_w_mean", SimpleImputer(strategy="mean")),
    ("scale_normal", StandardScaler())
])

pipeline_categorical = Pipeline([
    ("impute_w_most_frequent", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encode", OneHotEncoder(handle_unknown='ignore', sparse=False))
])

numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_cols = ['species', 'island']

preprocessing_transformer = ColumnTransformer([
    ("numeric", pipeline_numeric, numeric_cols),
    ("categorical", pipeline_categorical, categorical_cols)
])
```

---

## A Simple Model

Let's make a simple model for this data.  A Random Forest might be a nice one, let's try that.


```python
rf_clf = RandomForestClassifier()
```

Because our classifier has a ``fit``/``transform`` method, it can also be pipelined.  Let's take our _entire preprocessing transformer_ and make that the first step, then push that into the random forest classifier.


```python
preprocess_model_pipeline = Pipeline([
    ("preprocessing", preprocessing_transformer),
    ("random_forest_classifier", rf_clf)
])
```

---

## Time to Train

At this point, we'll break our original data into a training and test set and pass the training set through our pipeline.  Then we'll evaluate how we did!


```python
# Set up the Data.

df_features, df_target = get_and_clean_penguin_data()

x_train, x_test, y_train, y_test = train_test_split(
    df_features,
    df_target,
    test_size=0.33,
    random_state=1234
)

pmp = preprocess_model_pipeline.fit(x_train, y_train)

# Predict!
y_predicted = pmp.predict(x_test)

# Score!
scores = np.array([
    ("accuracy", accuracy_score(y_test, y_predicted)),
    ("precision", precision_score(y_test, y_predicted)),
    ("recall", recall_score(y_test, y_predicted)),
    ("f1", f1_score(y_test, y_predicted)),
])
df_scores = pd.DataFrame(scores[:, 1], index=scores[:, 0], columns=["value"])
df_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy</th>
      <td>0.9454545454545454</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.9811320754716981</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.9122807017543859</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.9454545454545454</td>
    </tr>
  </tbody>
</table>
</div>



Not too bad!

---

## The Complete Code.

Here's the code in one big chunk:


```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline


def get_and_clean_penguin_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Get and clean ``Penguins`` data.

    Loads ``Penguins`` data, removes rows with null values for ``sex``.
    Returns (df_features, df_target) as a tuple of dataframes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
    """

    df = pd.DataFrame(sns.load_dataset("penguins"))  # type: ignore
    df.dropna(subset=["sex"], inplace=True)

    # Transform Male/Female into 0/1.
    targets: pd.Series = (
        df["sex"].apply(lambda x: 0 if x == "Male" else 1) # type: ignore
    )

    return (df.drop("sex", axis=1), targets)


# PREPROCESSING PIPELINES
pipeline_numeric = Pipeline([
    ("impute_w_mean", SimpleImputer(strategy="mean")),
    ("scale_normal", StandardScaler())
])

pipeline_categorical = Pipeline([
    ("impute_w_most_frequent", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encode", OneHotEncoder(handle_unknown='ignore', sparse=False))
])

numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_cols = ['species', 'island']

preprocessing_transformer = ColumnTransformer([
    ("numeric", pipeline_numeric, numeric_cols),
    ("categorical", pipeline_categorical, categorical_cols)
])

# MODEL PIPELINES
rf_clf = RandomForestClassifier()

preprocess_model_pipeline = Pipeline([
    ("preprocessing", preprocessing_transformer),
    ("random_forest_classifier", rf_clf)
])

# TRAINING AND SCORING
df_features, df_target = get_and_clean_penguin_data()

x_train, x_test, y_train, y_test = train_test_split(
    df_features,
    df_target,
    test_size=0.33,
    random_state=1234
)

pmp = preprocess_model_pipeline.fit(x_train, y_train)
y_predicted = pmp.predict(x_test)

scores = np.array([
    ("accuracy", accuracy_score(y_test, y_predicted)),
    ("precision", precision_score(y_test, y_predicted)),
    ("recall", recall_score(y_test, y_predicted)),
    ("f1", f1_score(y_test, y_predicted)),
])
df_scores = pd.DataFrame(scores[:, 1], index=scores[:, 0], columns=["value"])
df_scores
```

Note that, with this structure, we could make different pipeline "pieces" to try out different classifiers, different params, etc.  The code is still a bit messy but for EDA it's able to be read through easily and able to be modified as needed with minimal difficulty.
