---
title:  "What parameters should I be using in my Sklearn Mmdels?: Introducing GridSearchCV."
date:   2022-01-08

classes: wide

header:
  overlay_filter: rgba(0, 146, 202, 0.8)
  overlay_image: /assets/images/title_grid.png
  caption: "Photo Credit: [**Wikipedia**](https://commons.wikimedia.org/wiki/File:Log_paper.svg)"
---
## Introduction

There are many different models to choose from in ``sklean`` to model your data with.  There are many parameters and hyper-parameters related to these models.  How can you find the **best** (or, at least, pretty good) ones for your data?


```python
import altair as alt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

---
## Grid Search

If you were posed this question and you didn't know a whole lot about the ``sklearn`` universe, you might say something like this:

"For every (hyper)parameter, let's take a list of values to try and ``for``-loop over them all."

That's pretty much hitting the nail on the head. Instead of doing ugly ``for``-loops some number of times (potentially indenting past what your monitor can show!), ``sklean`` has ``GridSearchCV``.  

Let's give an example, then chat about it.  If you've not read about [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), check out [this post](/2021-01-08-sklearn-pipelines-how-to.html).


```python
# Sample data, the Sklearn Digits Dataset.
df_features, df_targets = load_digits(return_X_y=True, as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(
    df_features, df_targets, train_size=0.33, random_state=1234
)

# Create our pipelines: Preprocess, Model.
pipeline_preprocess = Pipeline([("pca", PCA(n_components=3))])

pipeline_model = Pipeline([("random_forest", RandomForestClassifier(n_estimators=100))])

# Hook up our piplines together and train.
pipeline_full = Pipeline(
    [("preprocessing", pipeline_preprocess), ("modeling", pipeline_model)]
)

pipeline_full.fit(x_train, y_train)

# Score our model.
pipeline_full.score(x_test, y_test)
```




    0.7491694352159468



As we can see, we've made pipelines for preprocessing, modeling, and then tying those together.  It might seem verbose, but it makes things much easier when attempting to extend one part of the model, or swap things out.

While not a perfect model, it gets a respectable accuracy when running with the default parameters in ``PCA`` and ``RandomForestClassifier``.  Maybe tweaking these values would give a better result.  Suppose we try out something like, ``[1, 5, 10, 15, 20, 25, 30, 35]`` for the components in ``PCA``and ``[1, 10, 25, 50, 75, 100, 125]`` for ``n_estimators`` in the Random Forest &mdash; if you tried to do this yourself, you'd have to manually type in these values and run the model **56 times**.  That's much too much.  Instead, let's let grid-search do it for us.

(Note that, in addition to grid-searching, ``GridSearchCV`` will work on cross-validation scoring, so we no longer need to split our data into a train-test set.  However, we will rename and use the test set as the validation set at the end to score our model.)




```python
# Sample data, the Sklearn Digits Dataset.
df_features, df_targets = load_digits(return_X_y=True, as_frame=True)
x_train, x_validation, y_train, y_validation = train_test_split(
    df_features, df_targets, train_size=0.33, random_state=1234
)

# Create our pipelines: Preprocess, Model.
pipeline_preprocess = Pipeline([("pca", PCA(n_components=3))])

pipeline_model = Pipeline([("random_forest", RandomForestClassifier(n_estimators=100))])

# Hook up our piplines together and train.
pipeline_full = Pipeline(
    [("preprocessing", pipeline_preprocess), ("modeling", pipeline_model)]
)

# Parameters we're making a grid of.
#
# In our case, since `pipeline_full` is a pipeline of pipelines, we must
# use (pipeline_name)__(estimator_name)__(param_name).
#
# For example, `n_estimators` is given by `modeling__random_forest__n_estimators`.
#
# If you're not sure what to use, you can always print ``pipeline_full.get_params()``.
#
# See: https://scikit-learn.org/stable/modules/compose.html#nested-parameters

param_grid = {
    "modeling__random_forest__n_estimators": [1, 10, 25, 50, 75, 100, 125],
    "preprocessing__pca__n_components": [1, 5, 10, 15, 20, 25, 30, 35],
}

# NOTE: This takes about a minute.
grid_search = GridSearchCV(pipeline_full, param_grid)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(estimator=Pipeline(steps=[('preprocessing',
                                            Pipeline(steps=[('pca',
                                                             PCA(n_components=3))])),
                                           ('modeling',
                                            Pipeline(steps=[('random_forest',
                                                             RandomForestClassifier())]))]),
                 param_grid={'modeling__random_forest__n_estimators': [1, 10, 25,
                                                                       50, 75, 100,
                                                                       125],
                             'preprocessing__pca__n_components': [1, 5, 10, 15, 20,
                                                                  25, 30, 35]})




```python
print(grid_search.score(x_validation, y_validation))
print(grid_search.best_score_)
print(grid_search.best_estimator_)
```

    0.9493355481727574
    0.9628827802307363
    Pipeline(steps=[('preprocessing',
                     Pipeline(steps=[('pca', PCA(n_components=30))])),
                    ('modeling',
                     Pipeline(steps=[('random_forest',
                                      RandomForestClassifier(n_estimators=125))]))])


On the digits dataset, using more of the data via components and estimators gave us a better accuracy; this isn't always the case, and it's a good reason we grid-search in the first place.

One thing you might have noticed: this took a while to run.  Modeling the digit dataset is typically do-able in a second or less, but this took around a minute!  This seems trivial until we think about training datasets much larger than the digit dataset.  That's a problem.  This can be resolved in a few ways:

- Reduce your parameter space (using commonly accepted "good" parameters may work well!),
- Use a smarter grid-search (there are several out there which are a bit more complicated and situational),
- Try a bunch of different, spread-out parameters to try to hone in on areas which may be worth looking at,
- Trying something like ``RandomizedSearchCV``

There are many, many other potential solutions for the problem of "too big of a grid", but we will note one other thing here.  For parameters like regularization (which are commonly gridded), the workload can be reduced by computing the [regularization path](https://scikit-learn.org/stable/modules/grid_search.html#grid-search-tips).

It may also be worth checking out parallelization methods if you're going to be using larger grids on significant amounts of data.

---
## What do we get from GridSearchCV?

When we ran ``GridSearchCV`` above, we took the ``.best_estimator_`` and were done with it. Can we look a bit closer into the results?  Sure.


```python
df_grid_search_results = pd.DataFrame(grid_search.cv_results_)
df_grid_search_results = df_grid_search_results[
    [
        "param_modeling__random_forest__n_estimators",
        "param_preprocessing__pca__n_components",
        "mean_test_score",
    ]
]
df_grid_search_results.head(5)
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
      <th>param_modeling__random_forest__n_estimators</th>
      <th>param_preprocessing__pca__n_components</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.227603</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>0.741988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10</td>
      <td>0.704885</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>15</td>
      <td>0.684475</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>20</td>
      <td>0.709927</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot these values.
chart = (
    alt.Chart(df_grid_search_results)
    .encode(
        x="param_modeling__random_forest__n_estimators:Q",
        y="param_preprocessing__pca__n_components",
        color=alt.Color("mean_test_score", scale=alt.Scale(scheme="redblue")),
    )
    .configure_axis(grid=False)
    .mark_circle()
)
chart
```





<div id="altair-viz-416be54abd3f408c849fc1b369a998c6"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-416be54abd3f408c849fc1b369a998c6") {
      outputDiv = document.getElementById("altair-viz-416be54abd3f408c849fc1b369a998c6");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"grid": false}}, "data": {"name": "data-e3bbf9eb237ea0b748cdcfe5e0281ff9"}, "mark": "circle", "encoding": {"color": {"field": "mean_test_score", "scale": {"scheme": "redblue"}, "type": "quantitative"}, "x": {"field": "param_modeling__random_forest__n_estimators", "type": "quantitative"}, "y": {"field": "param_preprocessing__pca__n_components", "type": "quantitative"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-e3bbf9eb237ea0b748cdcfe5e0281ff9": [{"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.22760290556900725}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.7419883207520297}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.7048853439680958}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.6844751459905997}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.7099273607748183}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.6543227460475716}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.6390542657741063}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.605440820395955}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.227617148554337}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8735080472867113}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.8971229169633956}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9173052271756161}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9021364477994588}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9089161088164079}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.8835920809001566}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.8768693918245264}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.22931206380857425}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8902720410197977}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9257513174761429}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9459905996296823}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9358780800455776}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9375872382851445}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9358923230309072}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9443241703461046}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.2309927360774818}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8970944309927361}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9341404358353509}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9442814413901154}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9460333285856715}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9443241703461045}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9493804301381568}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9460190856003419}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.2309927360774818}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.900413046574562}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9324882495371029}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9476997578692492}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.947714000854579}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9459905996296823}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9426434980771969}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9493946731234868}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.2309927360774818}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.9088448938897592}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9291269049992877}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9476712718985899}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9611736219911693}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9510611024070645}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9561458481697764}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9510468594217347}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.2309927360774818}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8987751032616437}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9392251815980629}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9510611024070645}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9578122774533542}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9510753453923941}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9628827802307363}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.957826520438684}]}}, {"mode": "vega-lite"});
</script>



Seems like a lot of the grid was pretty close, in terms of scoring.  This is reasonable, given how small and simple the data is.

---
## One More Example

Let's do one last easy example to solidify this.  We'll use the iris dataset, but we'll have a ridiculously small training size.  Let's see how well we can do.


```python
df_features, df_targets = load_iris(return_X_y=True, as_frame=True)
x_train, x_validation, y_train, y_validation = train_test_split(
    df_features, df_targets, train_size=0.15, random_state=1234
)

pipeline_preprocess = Pipeline(
    [("scaler", StandardScaler()), ("pca", PCA(n_components=3))]
)

# NOTE: Here, we could have used LogisticRegressionCV to grid values for C.
# Since we're focusing on GridSearchCV, I decided to use the standard LogReg.
pipeline_model = Pipeline(
    [("logistic_regression", LogisticRegression(C=1.0, max_iter=1_000))]
)

# Hook up our piplines together and train.
pipeline_full = Pipeline(
    [("preprocessing", pipeline_preprocess), ("modeling", pipeline_model)]
)

param_grid = {
    "modeling__logistic_regression__C": np.logspace(-4, 2, 10),
    "preprocessing__pca__n_components": [1, 2, 3, 4],
}

# NOTE: Takes a few seconds.
grid_search = GridSearchCV(pipeline_full, param_grid)
grid_search.fit(x_train, y_train)
```




    GridSearchCV(estimator=Pipeline(steps=[('preprocessing',
                                            Pipeline(steps=[('scaler',
                                                             StandardScaler()),
                                                            ('pca',
                                                             PCA(n_components=3))])),
                                           ('modeling',
                                            Pipeline(steps=[('logistic_regression',
                                                             LogisticRegression(max_iter=1000))]))]),
                 param_grid={'modeling__logistic_regression__C': array([1.00000000e-04, 4.64158883e-04, 2.15443469e-03, 1.00000000e-02,
           4.64158883e-02, 2.15443469e-01, 1.00000000e+00, 4.64158883e+00,
           2.15443469e+01, 1.00000000e+02]),
                             'preprocessing__pca__n_components': [1, 2, 3, 4]})




```python
df_grid_search_results = pd.DataFrame(grid_search.cv_results_)
df_grid_search_results = df_grid_search_results[
    [
        "param_modeling__logistic_regression__C",
        "param_preprocessing__pca__n_components",
        "mean_test_score",
    ]
]

chart = (
    alt.Chart(df_grid_search_results)
    .encode(
        x="param_modeling__logistic_regression__C:Q",
        y="param_preprocessing__pca__n_components",
        color=alt.Color("mean_test_score", scale=alt.Scale(scheme="redblue")),
    )
    .configure_axis(grid=False)
    .mark_circle()
)
chart
```





<div id="altair-viz-d522c821050c4f74abd7685bf7dfbdc0"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-d522c821050c4f74abd7685bf7dfbdc0") {
      outputDiv = document.getElementById("altair-viz-d522c821050c4f74abd7685bf7dfbdc0");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"grid": false}}, "data": {"name": "data-49d735abdf212efc3b4bcafcc3fafd56"}, "mark": "circle", "encoding": {"color": {"field": "mean_test_score", "scale": {"scheme": "redblue"}, "type": "quantitative"}, "x": {"field": "param_modeling__logistic_regression__C", "type": "quantitative"}, "y": {"field": "param_preprocessing__pca__n_components", "type": "quantitative"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-49d735abdf212efc3b4bcafcc3fafd56": [{"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.77}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.77}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.77}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 3, "mean_test_score": 1.0}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 4, "mean_test_score": 1.0}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 3, "mean_test_score": 1.0}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 4, "mean_test_score": 1.0}]}}, {"mode": "vega-lite"});
</script>




```python
print(grid_search.best_params_)
print(grid_search.best_score_)
```

    {'modeling__logistic_regression__C': 21.54434690031882, 'preprocessing__pca__n_components': 3}
    1.0


Interesting!  Of course, this isn't meant to show the best models for these smaller datasets, but rather how to use the tools for your larger, more complex data.

_Happy gridding!_  ``:']``
