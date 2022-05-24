---
title:  "Merging Dictionaries in Python 3.9"
date:   2022-05-24

description: Quick and easy method to merge dictionaries in Python 3.9+.
categories: python linting

excerpt: ""

classes: wide

header:
  overlay_filter: rgba(0, 146, 202, 0.8)
  overlay_image: /assets/images/title-merge-dictionaries.jpg
  caption: "Photo Credit: [**Wiki**](https://en.wikipedia.org/wiki/File:Escribano.jpg)"
---
# Introduction

Here's a problem that comes up a bunch in Python.  Suppose we have two dictionaries:

```python
dict_1 = {"jimmy": 1, "jane": 4}
dict_2 = {"billy": 23, "beth": 10}
```
How do we make this into a single dictionary?

_Note here that there are considerations: What if the same key is in both dictionaries?  What if they have different types for keys?  Etc, etc.  Having said that, we're going to assume that everything is nice and that we know that everything will be okay if we merge these._


```python
# The old way.
dict_1 = {"jimmy": 1, "jane": 4}
dict_2 = {"billy": 23, "beth": 10}

dict_merged = {**dict_1, **dict_2}
dict_merged

```




    {'jimmy': 1, 'jane': 4, 'billy': 23, 'beth': 10}




```python
# The new way, Python 3.9+.
dict_1 = {"jimmy": 1, "jane": 4}
dict_2 = {"billy": 23, "beth": 10}

dict_merged = dict_1 | dict_2
dict_merged
```




    {'jimmy': 1, 'jane': 4, 'billy': 23, 'beth': 10}


This feels kind of strange ("Why the 'or' symbol?") unless we think of these more as sets of keys and their associated values: in this case, we are using a legit set union (an "or").

For more information on this change, see the corresponding [PEP-0584](https://peps.python.org/pep-0584/).
