---
title:  "GridSearchCV: What Should My Parameter Be?"
date:   2022-01-08

description: The basics of Sklearn grid searching.
categories: python sklearn datascience data

excerpt: There are many different models to choose from in ``sklean`` to model your data with.  There are many parameters and hyper-parameters related to these models.  How can you find the best ones for your data?

classes: wide
---

## Introduction

There are many different models to choose from in ``sklean`` to model your data with.  There are many parameters and hyper-parameters related to these models.  How can you find the best ones for your data?

The short answer is: _you can't_.

Nevertheless, we can try our best at getting something pretty good.  Below, let's import everything we need.  Don't worry if you don't know what all of this is yet!


```python
import altair as alt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    df_features,
    df_targets,
    train_size=0.33,
    random_state=1234
)

# Create our pipelines: Preprocess, Model.
pipeline_preprocess = Pipeline([
    ("pca", PCA(n_components=3))
])

pipeline_model = Pipeline([
    ("random_forest", RandomForestClassifier(n_estimators=100))
])

# Hook up our piplines together and train.
pipeline_full = Pipeline([
    ("preprocessing", pipeline_preprocess),
    ("modeling", pipeline_model)
])

pipeline_full.fit(x_train, y_train)

# Score our model.
pipeline_full.score(x_test, y_test)

```




    0.7416943521594684



As we can see, we've made pipelines for preprocessing, modeling, and then tying those together.  It might seem verbose, but it makes things much easier when attempting to extend one part of the model, or swap things out.

While not a perfect model, it gets a respectable accuracy when running with the default parameters in ``PCA`` and ``RandomForestClassifier``.  Maybe tweaking these values would give a better result.  Suppose we try out something like, ``[1, 5, 10, 15, 20, 25, 30, 35]`` for the components in ``PCA``and ``[1, 10, 25, 50, 75, 100, 125]`` for ``n_estimators`` in the Random Forest &mdash; if you tried to do this yourself, you'd have to manually type in these values and run the model **56 times**.  That's much too much.  Instead, let's let grid-search do it for us.

(Note that, in addition to grid-searching, ``GridSearchCV`` will work on cross-validation scoring, so we no longer need to split our data into a train-test set.  However, we will rename and use the test set as the validation set at the end to score our model.)




```python
# Sample data, the Sklearn Digits Dataset.
df_features, df_targets = load_digits(return_X_y=True, as_frame=True)
x_train, x_validation, y_train, y_validation = train_test_split(
    df_features,
    df_targets,
    train_size=0.33,
    random_state=1234
)

# Create our pipelines: Preprocess, Model.
pipeline_preprocess = Pipeline([
    ("pca", PCA(n_components=3))
])

pipeline_model = Pipeline([
    ("random_forest", RandomForestClassifier(n_estimators=100))
])

# Hook up our piplines together and train.
pipeline_full = Pipeline([
    ("preprocessing", pipeline_preprocess),
    ("modeling", pipeline_model)
])

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
    "preprocessing__pca__n_components": [1, 5, 10, 15, 20, 25, 30, 35]
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

    0.9476268412438625
    0.9696592561075063
    Pipeline(steps=[('preprocessing',
                     Pipeline(steps=[('pca', PCA(n_components=35))])),
                    ('modeling',
                     Pipeline(steps=[('random_forest',
                                      RandomForestClassifier())]))])


On the digits dataset, using more of the data via components and estimators gave us a better accuracy; this isn't always the case, and it's a good reason we grid-search in the first place.

One thing you might have noticed: this took a while to run.  Modeling the digit dataset is typically do-able in a second or less, but this took around a minute!  This seems trivial until we think about training datasets much larger than the digit dataset.  That's a problem.  This can be resolved in a few ways:

- Reduce your parameter space (using commonly accepted "good" parameters may work well!),
- Use a smarter grid-search (there are several out there which are a bit more complicated and situational),
- Try a bunch of different, spread-out parameters to try to hone in on areas which may be worth looking at,
- Trying something like ``RandomizedSearchCV``

There are many, many other potential solutions for the problem of "too big of a grid", but we will note one other thing here.  For parameters like regularization (which are commonly gridded), the workload can be reduced by computing the (regularization path)[https://scikit-learn.org/stable/modules/grid_search.html#grid-search-tips].

It may also be worth checking out parallelization methods if you're going to be using larger grids on significant amounts of data.

---
## What do we get from GridSearchCV?

When we ran ``GridSearchCV`` above, we took the ``.best_estimator_`` and were done with it. Can we look a bit closer into the results?  Sure.


```python
df_grid_search_results = pd.DataFrame(grid_search.cv_results_)
df_grid_search_results = df_grid_search_results[[
    "param_modeling__random_forest__n_estimators",
    "param_preprocessing__pca__n_components",
    "mean_test_score",
]]
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
      <td>0.298451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>0.774003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10</td>
      <td>0.779952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>15</td>
      <td>0.743644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>20</td>
      <td>0.710804</td>
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
            color=alt.Color("mean_test_score", scale=alt.Scale(scheme='redblue'))
        )
        .configure_axis(grid=False)
        .mark_circle()

)
chart.interactive()
```





<div id="altair-viz-5c55731c8ea64eacb0ffe71f17e16420"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-5c55731c8ea64eacb0ffe71f17e16420") {
      outputDiv = document.getElementById("altair-viz-5c55731c8ea64eacb0ffe71f17e16420");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"grid": false}}, "data": {"name": "data-5f304604ed718d9e023652f934b7c47d"}, "mark": "circle", "encoding": {"color": {"field": "mean_test_score", "scale": {"scheme": "redblue"}, "type": "quantitative"}, "x": {"field": "param_modeling__random_forest__n_estimators", "type": "quantitative"}, "y": {"field": "param_preprocessing__pca__n_components", "type": "quantitative"}}, "selection": {"selector015": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-5f304604ed718d9e023652f934b7c47d": [{"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.2984505194482857}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.7740027656632273}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.7799524873240435}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.7436442931603021}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.7108038151969648}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.7049072793674431}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.7132964578236358}, {"param_modeling__random_forest__n_estimators": 1, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.6315462894018367}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.30519802857851996}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8904088217565507}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9106407119809949}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9232918483849236}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9232705740524059}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9199198666808496}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9207424742048719}, {"param_modeling__random_forest__n_estimators": 10, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9114775023933624}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.3051838456901748}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8979931212991525}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9393114207708401}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9502641562954295}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.951944828564337}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9443711661879941}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9485763925823493}, {"param_modeling__random_forest__n_estimators": 25, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.940165939793639}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.3043399638336347}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8996773392901464}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9334077934971458}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9528028933092225}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9553345388788426}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9528064390313087}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9586994291387441}, {"param_modeling__random_forest__n_estimators": 50, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9595326738290254}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.30434350955572104}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.9089635854341737}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9376201113356736}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9553416303230152}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9528064390313087}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9629082012551857}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9646030564124384}, {"param_modeling__random_forest__n_estimators": 75, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9603836471297381}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.3051838456901748}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.9055774208417546}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9443640747438217}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9570293940360954}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.957866184448463}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9595397652731978}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9645959649682657}, {"param_modeling__random_forest__n_estimators": 100, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9696592561075063}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.3060241818246286}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 5, "mean_test_score": 0.8971598766088714}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 10, "mean_test_score": 0.9418466120625466}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 15, "mean_test_score": 0.9544906570223025}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 20, "mean_test_score": 0.9595468567173706}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 25, "mean_test_score": 0.9629117469772719}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 30, "mean_test_score": 0.9671276105378862}, {"param_modeling__random_forest__n_estimators": 125, "param_preprocessing__pca__n_components": 35, "mean_test_score": 0.9595468567173704}]}}, {"mode": "vega-lite"});
</script>



Seems like a lot of the grid was pretty close, in terms of scoring.  This is reasonable, given how small and simple the data is.

---
## One More Example

Let's do one last easy example to solidify this.  We'll use the iris dataset, but we'll have a ridiculously small training size.  Let's see how well we can do.


```python
df_features, df_targets = load_iris(return_X_y=True, as_frame=True)
x_train, x_validation, y_train, y_validation = train_test_split(
    df_features,
    df_targets,
    train_size=0.15,
    random_state=1234
)

pipeline_preprocess = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=3))
])

# NOTE: Here, we could have used LogisticRegressionCV to grid values for C.
# Since we're focusing on GridSearchCV, I decided to use the standard LogReg.
pipeline_model = Pipeline([
    ("logistic_regression", LogisticRegression(C=1.0, max_iter=1_000))
])

# Hook up our piplines together and train.
pipeline_full = Pipeline([
    ("preprocessing", pipeline_preprocess),
    ("modeling", pipeline_model)
])

param_grid = {
    "modeling__logistic_regression__C": np.logspace(-4, 2, 10),
    "preprocessing__pca__n_components": [1, 2, 3, 4]
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
df_grid_search_results = df_grid_search_results[[
    "param_modeling__logistic_regression__C",
    "param_preprocessing__pca__n_components",
    "mean_test_score",
]]

chart = (
    alt.Chart(df_grid_search_results)
        .encode(
            x="param_modeling__logistic_regression__C:Q",
            y="param_preprocessing__pca__n_components",
            color=alt.Color("mean_test_score", scale=alt.Scale(scheme='redblue'))
        )
        .configure_axis(grid=False)
        .mark_circle()

)
chart.interactive()
```





<div id="altair-viz-390d5918e4514cf3a30c8aadf6f1f2a2"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-390d5918e4514cf3a30c8aadf6f1f2a2") {
      outputDiv = document.getElementById("altair-viz-390d5918e4514cf3a30c8aadf6f1f2a2");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"grid": false}}, "data": {"name": "data-49d735abdf212efc3b4bcafcc3fafd56"}, "mark": "circle", "encoding": {"color": {"field": "mean_test_score", "scale": {"scheme": "redblue"}, "type": "quantitative"}, "x": {"field": "param_modeling__logistic_regression__C", "type": "quantitative"}, "y": {"field": "param_preprocessing__pca__n_components", "type": "quantitative"}}, "selection": {"selector017": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-49d735abdf212efc3b4bcafcc3fafd56": [{"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.0001, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.00046415888336127773, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.002154434690031882, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.41}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.01, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.77}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.77}, {"param_modeling__logistic_regression__C": 0.046415888336127774, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.77}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.72}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 0.21544346900318823, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 1.0, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 3, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 4.641588833612772, "param_preprocessing__pca__n_components": 4, "mean_test_score": 0.95}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 3, "mean_test_score": 1.0}, {"param_modeling__logistic_regression__C": 21.54434690031882, "param_preprocessing__pca__n_components": 4, "mean_test_score": 1.0}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 1, "mean_test_score": 0.9099999999999999}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 2, "mean_test_score": 0.8699999999999999}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 3, "mean_test_score": 1.0}, {"param_modeling__logistic_regression__C": 100.0, "param_preprocessing__pca__n_components": 4, "mean_test_score": 1.0}]}}, {"mode": "vega-lite"});
</script>




```python
print(grid_search.best_params_)
print(grid_search.best_score_)
```

    {'modeling__logistic_regression__C': 21.54434690031882, 'preprocessing__pca__n_components': 3}
    1.0


Interesting!  Of course, this isn't meant to show the best models for these smaller datasets, but rather how to use the tools for your larger, more complex data.

_Happy gridding!_  ``:']``
