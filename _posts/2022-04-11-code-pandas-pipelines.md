---
title:  "Using Pipes in Pandas for Fun and Profit"
date:   2022-04-11

classes: wide

header:
  overlay_filter: rgba(0, 146, 202, 0.8)
  overlay_image: /assets/images/title_pipes_in_yard.jpg
  caption: "Photo Credit: [**Bill of sale-AO 3765**](https://commons.wikimedia.org/wiki/File:Bill_of_sale_Louvre_AO3765.jpg)"
---
# Introduction

One of my pet peeves when looking at data scientist code is how _disorganized_ it tends to be &mdash; my own included!  Because we're experimenting, looking at a variety of data in different ways, and shift+enter-ing in our notebooks like there's no tomorrow, it's difficult to figure out how to make things look organized.

One thing that's especially vexing is the following problem:

_I have a bunch of things I'm doing to the raw data to clean it up.  How should I organize these things?_

I've seen this done many, many different ways: a single class, many classes, single function, many functions, many commands, nested commands, chained commands, dropping to SQL and doing the manipulations there, ...

In this post, we'll go over a particularly elegant solution.  Using Pipes, we can define small functions and "pipe" them together, making the processing obvious and well-labeled.  

**In general, we can use pipes for dataframes whenever we have functions where the ouput of one of the functions is the input for another.  In this way, piping allows us to chain these functions.**

Let's check it out for a toy example.


```python
import pandas as pd
import numpy as np
```


```python
def raw_data(size=1000) -> pd.DataFrame:
    """Makes a random, raw dataset for us to use."""
    age = np.random.randint(18, 80, size=size)
    salary = np.array([f"${np.random.randint(30000, 190000)}" for _ in range(size)])
    department = np.random.choice(["A", "B", "C"], size=size)
    title = np.random.choice(["1a", "1b", "2a"], size=size)
    is_emplyed = np.random.choice(["Yes", "No"], size=size)

    data = {
        "age": age,
        "salary": salary,
        "department": department,
        "title": title,
        "is_employed": is_emplyed,
    }
    return pd.DataFrame(data)
```


```python
df_raw = raw_data()
df_raw.head()
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
      <th>age</th>
      <th>salary</th>
      <th>department</th>
      <th>title</th>
      <th>is_employed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>$67824</td>
      <td>C</td>
      <td>1a</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>$31265</td>
      <td>A</td>
      <td>2a</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>$38033</td>
      <td>A</td>
      <td>1a</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>$135436</td>
      <td>C</td>
      <td>2a</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>$130890</td>
      <td>B</td>
      <td>2a</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



We have a toy dataset above.  We have the age of the person, their salary, their "department" (A, B, or C), their "title" (1a, 1b, or 2a), and if they are currently employeed or not.

Let's say we want to do the following operations:

- Bin the age values in ten year bins (20-30, 30-40, 40-50, and so forth...),
- Parse the salary so that it's an int value,
- Concatenate the department and title so it looks like "C - 1b",
- Convert the is_employed column to boolean,
- Remove any employee which is not employeed currenty (is_employed is "No").

Let's try this stuff out!


```python
# Binning the ages.


def bin_ages(df: pd.DataFrame) -> str:
    """Bins the col_name column in slices of 10."""
    cut_bins = [10 * i for i in range(10)]  # [0, 10, 20, ...]
    cut_labels = [
        f"{cut_bins[idx]} -- {cut_bins[idx + 1]}" for idx in range(len(cut_bins) - 1)
    ]
    df[f"age_binned"] = pd.cut(df["age"], bins=cut_bins, labels=cut_labels)

    return df
```


```python
bin_ages(df_raw).head(5)
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
      <th>age</th>
      <th>salary</th>
      <th>department</th>
      <th>title</th>
      <th>is_employed</th>
      <th>age_binned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>$67824</td>
      <td>C</td>
      <td>1a</td>
      <td>No</td>
      <td>20 -- 30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>$31265</td>
      <td>A</td>
      <td>2a</td>
      <td>No</td>
      <td>60 -- 70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>$38033</td>
      <td>A</td>
      <td>1a</td>
      <td>Yes</td>
      <td>20 -- 30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>$135436</td>
      <td>C</td>
      <td>2a</td>
      <td>No</td>
      <td>20 -- 30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>$130890</td>
      <td>B</td>
      <td>2a</td>
      <td>Yes</td>
      <td>50 -- 60</td>
    </tr>
  </tbody>
</table>
</div>



Neat.  Maybe not the most efficient function, but it works.  **Note that the first parameter of this function will be the "input" variable for our pipe, so we make it the dataframe.**  We'll see what this means in a little bit.

Note also that we're **overwriting the original dataframe** in these functions.  You may want to have an argument which copies the dataframe when debugging these functions, mainly for brevity.


```python
# Salary is an int value.  Note that we're excluding a lot of error-checking
# for the sake of simplicity here.


def parse_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Parses a salary value like $10000 into an int."""
    df["salary_int"] = df["salary"].apply(lambda x: int(x.replace("$", "")))
    return df
```


```python
parse_salary(df_raw).head(5)
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
      <th>age</th>
      <th>salary</th>
      <th>department</th>
      <th>title</th>
      <th>is_employed</th>
      <th>age_binned</th>
      <th>salary_int</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>$67824</td>
      <td>C</td>
      <td>1a</td>
      <td>No</td>
      <td>20 -- 30</td>
      <td>67824</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>$31265</td>
      <td>A</td>
      <td>2a</td>
      <td>No</td>
      <td>60 -- 70</td>
      <td>31265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>$38033</td>
      <td>A</td>
      <td>1a</td>
      <td>Yes</td>
      <td>20 -- 30</td>
      <td>38033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>$135436</td>
      <td>C</td>
      <td>2a</td>
      <td>No</td>
      <td>20 -- 30</td>
      <td>135436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>$130890</td>
      <td>B</td>
      <td>2a</td>
      <td>Yes</td>
      <td>50 -- 60</td>
      <td>130890</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Concatenate dept and title with --.


def concatenate_dept_and_title(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenates dept and title like "Dept -- Title"."""
    df["dept_title"] = df["department"].str.cat(df["title"], sep=" -- ")
    return df
```


```python
concatenate_dept_and_title(df_raw).head(5)
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
      <th>age</th>
      <th>salary</th>
      <th>department</th>
      <th>title</th>
      <th>is_employed</th>
      <th>age_binned</th>
      <th>salary_int</th>
      <th>dept_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>$67824</td>
      <td>C</td>
      <td>1a</td>
      <td>No</td>
      <td>20 -- 30</td>
      <td>67824</td>
      <td>C -- 1a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>$31265</td>
      <td>A</td>
      <td>2a</td>
      <td>No</td>
      <td>60 -- 70</td>
      <td>31265</td>
      <td>A -- 2a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>$38033</td>
      <td>A</td>
      <td>1a</td>
      <td>Yes</td>
      <td>20 -- 30</td>
      <td>38033</td>
      <td>A -- 1a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>$135436</td>
      <td>C</td>
      <td>2a</td>
      <td>No</td>
      <td>20 -- 30</td>
      <td>135436</td>
      <td>C -- 2a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>$130890</td>
      <td>B</td>
      <td>2a</td>
      <td>Yes</td>
      <td>50 -- 60</td>
      <td>130890</td>
      <td>B -- 2a</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Converting the employed col to boolean.
# Note again we're not doing a lot of error checking for
# simplicity; you ought to check that no other values exist here!


def convert_employed_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    """Converts employed col to boolean."""
    df["is_employed_bool"] = df["is_employed"].apply(lambda x: x == "Yes")
    return df
```


```python
convert_employed_to_bool(df_raw).head(5)
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
      <th>age</th>
      <th>salary</th>
      <th>department</th>
      <th>title</th>
      <th>is_employed</th>
      <th>age_binned</th>
      <th>salary_int</th>
      <th>dept_title</th>
      <th>is_employed_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>$67824</td>
      <td>C</td>
      <td>1a</td>
      <td>No</td>
      <td>20 -- 30</td>
      <td>67824</td>
      <td>C -- 1a</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>$31265</td>
      <td>A</td>
      <td>2a</td>
      <td>No</td>
      <td>60 -- 70</td>
      <td>31265</td>
      <td>A -- 2a</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>$38033</td>
      <td>A</td>
      <td>1a</td>
      <td>Yes</td>
      <td>20 -- 30</td>
      <td>38033</td>
      <td>A -- 1a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>$135436</td>
      <td>C</td>
      <td>2a</td>
      <td>No</td>
      <td>20 -- 30</td>
      <td>135436</td>
      <td>C -- 2a</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>$130890</td>
      <td>B</td>
      <td>2a</td>
      <td>Yes</td>
      <td>50 -- 60</td>
      <td>130890</td>
      <td>B -- 2a</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dropping all individuals not employed.


def drop_not_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Drops all individuals who are not employed."""
    df = df[df["is_employed_bool"]]
    return df
```


```python
drop_not_employed(convert_employed_to_bool(df_raw)).head(5)
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
      <th>age</th>
      <th>salary</th>
      <th>department</th>
      <th>title</th>
      <th>is_employed</th>
      <th>age_binned</th>
      <th>salary_int</th>
      <th>dept_title</th>
      <th>is_employed_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>$38033</td>
      <td>A</td>
      <td>1a</td>
      <td>Yes</td>
      <td>20 -- 30</td>
      <td>38033</td>
      <td>A -- 1a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>$130890</td>
      <td>B</td>
      <td>2a</td>
      <td>Yes</td>
      <td>50 -- 60</td>
      <td>130890</td>
      <td>B -- 2a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>58</td>
      <td>$57380</td>
      <td>C</td>
      <td>2a</td>
      <td>Yes</td>
      <td>50 -- 60</td>
      <td>57380</td>
      <td>C -- 2a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>60</td>
      <td>$66386</td>
      <td>C</td>
      <td>1b</td>
      <td>Yes</td>
      <td>50 -- 60</td>
      <td>66386</td>
      <td>C -- 1b</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>46</td>
      <td>$181794</td>
      <td>B</td>
      <td>1b</td>
      <td>Yes</td>
      <td>40 -- 50</td>
      <td>181794</td>
      <td>B -- 1b</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop all of the old columns that we don't need anymore.
def drop_old_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnecessary, pre-cleaning columns."""
    df = df.drop(["age", "salary", "title", "department", "is_employed"], axis=1)
    return df
```

At this point we've got all our functions.  Let's see what they look like in a pipeline.


```python
df_cleaned = (
    df_raw.pipe(bin_ages)
    .pipe(parse_salary)
    .pipe(concatenate_dept_and_title)
    .pipe(convert_employed_to_bool)
    .pipe(drop_not_employed)
    .pipe(drop_old_columns)
)

df_cleaned.head(5)
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
      <th>age_binned</th>
      <th>salary_int</th>
      <th>dept_title</th>
      <th>is_employed_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20 -- 30</td>
      <td>38033</td>
      <td>A -- 1a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50 -- 60</td>
      <td>130890</td>
      <td>B -- 2a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>50 -- 60</td>
      <td>57380</td>
      <td>C -- 2a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>50 -- 60</td>
      <td>66386</td>
      <td>C -- 1b</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>40 -- 50</td>
      <td>181794</td>
      <td>B -- 1b</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Note that we could pass arguments to these functions, as demonstrated in the [docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html).

Having this pipeline allows us to see, at a glance, what we're doing to this data and if that is, in fact, what we want to do.  

Moreover, it is in a _single cell_ (for Jupyter Notebooks), which gets rid of that annoying "did I run this yet?" problem that I've seen a number of data scientists run into.  It is not a replacement for good notebook practices, but it does make it easier to get into good notebook habits!
