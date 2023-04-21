---
title:  "Linting Tidbit: Pylint and logging-format-interpolation"
date:   2022-03-28

classes: wide

header:
  overlay_filter: rgba(0, 146, 202, 0.8)
  overlay_image: /assets/images/title_logging_format_interpolation.jpg
  caption: "Photo Credit: [**Aapo Haapanen**](https://commons.wikimedia.org/wiki/File:Logs.jpg#file)"
---
## Introduction

While refactoring some code at work, Pylint produced a warning I found strange:

> Use lazy % formatting in logging functions [logging-fstring-interpolation]

Three things: 
1. What is the issue?
2. How serious is it?
3. How do we fix it?

## What is the issue?

The issue is easy to reproduce, so let's do so below:


```python
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

my_name = "James"
logging.info(f"Howdy, {my_name}!")
```

    INFO:root:Howdy, James!


Running this in a Python file with an IDE that has Pylint enabled will give the warning above.

What's the deal? We haven't done anything usual. We're logging using an f-string, and that's [the preferred method of text-formatting](https://peps.python.org/pep-0498/), right?

The rationale is written up [here](https://github.com/PyCQA/pylint/issues/2395) and a few other locations if you search for the warning.  The long and short of it is: evaluation of statements in messages is deferred until it needs to be done, and an f-string is _not_ deferred evaluation.  If you have an expensive thing that the log needs to compute, then that might give you a performance hit.


```python
def make_expensive_call():
    """Make an "expensive" call."""
    return "".join("" for _ in range(10_000_000)).strip()


# f-string method.
logging.info(f"{make_expensive_call()}")

# The pylint preferred method.
logging.info("%s" % make_expensive_call())
```

    INFO:root:
    INFO:root:


Note that this syntax is similar to the syntax used in SQLAlchemy, so it's not completely alien.  In addition to lazy evaluation, you also get some protection against injections in case, for whatever reason, you need to log out user input.

## How serious is it?

Unless you're doing some heavy-duty logging or some other pathological things, probably not very serious.  Additionally, in the author's opinion, the Pylint preferred method feels awkward to write and read.  For example,


```python
my_name = "James"
my_age = 100
my_job = "Computer"
my_dogs_name = "My Dog Friday"

# Pylint Preferred method.
logging.info(
    "Hi, I'm %s.  I'm %s years old, and I'm a %s.  My dog is named %s"
    % (my_name, my_age, my_job, my_dogs_name)
)

# f-string method.
logging.info(
    f"Hi, I'm {my_name}.  I'm {my_age} years old, and I'm a {my_job}.  "
    f"My dog is named {my_dogs_name}"
)
```

    INFO:root:Hi, I'm James.  I'm 100 years old, and I'm a Computer.  My dog is named My Dog Friday
    INFO:root:Hi, I'm James.  I'm 100 years old, and I'm a Computer.  My dog is named My Dog Friday


Of course, this is a matter of opinion, and will depend on the type of logging one does.  Either way, it does not feel like Pylint should force you to do one or the other &mdash;

Luckily, it does not.

## How do we fix it?

You can do one of two things:
- Write your logs as above in the Pylint Preferred way, with those %s things.
- As of [Pylint 2.5](https://pylint.pycqa.org/en/latest/whatsnew/2.5.html?highlight=fstring) there is a method to disable this message.  

For the latter, if you're using a ``pyproject.toml``, you can put the following configuration in the file to disable the warning:

```raw
[tool.pylint.LOGGING]
disable=["logging-fstring-interpolation"]
```

It is disabled similarly with other configs.


