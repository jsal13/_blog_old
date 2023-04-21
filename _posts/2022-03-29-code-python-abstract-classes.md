---
title:  "Abstract Classes In Python"
date:   2022-03-29

classes: wide
header:
  overlay_filter: rgba(0, 146, 202, 0.8)
  overlay_image: /assets/images/title_contract_abstract_classes.jpg
  caption: "Photo Credit: [**Bill of sale-AO 3765**](https://commons.wikimedia.org/wiki/File:Bill_of_sale_Louvre_AO3765.jpg)"
---
## Introduction

Abstract Classes may seem like strange beasts that one learns about and promptly forgets, but understanding what they are and when to use them will allow you to structure software in more readable, reliable, and scalable ways.

For this post, I'm going to take a fairly simplified approach and give an example of one of the ways abstract classes can be used.  Let's dive in.

## The Sample Problem

Let's say that we've got one database with all of our data on it.  Let's outline a little class for the connector. 

```python
class DBConnector:
    """Connector for our DB."""

    def __init__(
        self, 
        db: Any, 
        host: str, 
        user: str, 
        password: str
    ) -> None:
        """Your amazing docstring goes here."""
        self.db = db
        self.host = host
        self.user = user
        self.password = password

    def _connect(self) -> Any:  # Connection type can go here in place of `Any`
        # Do all of your connection stuff here...
        conn = ...
        # ...
        # ...
        return conn

    def query(self, query: str) -> List[Tuple[str]]:
        """
        Run the query.  
        
        Note this connection is a mock, you'll have to replace it 
        with whatever connection object you're using.
        """
        conn = self._connect()
        return conn.run_a_query(query)
```

Pretty basic.  You'd run this in the following way:

```python
# Put the args into DBConnector.
dbconn = DBConnector(...).query("SELECT * FROM my_table")
```

This would return some data in a reasonable way.

---

### Scaling the Connector...
**What happens when someone wants to add another type of DB with a different connector?**  This becomes a bigger problem.  Maybe you try to patch it like this:

```python
from typing import Any, List, Tuple

class DBConnector:
    """Connector for our DB."""

    def __init__(
        self, 
        db: Any, 
        host: str, 
        user: str, 
        password: str, 
        db_type: str
    ) -> None:
        """Your amazing docstring goes here."""
        self.db = db
        self.host = host
        self.user = user
        self.password = password
        self.db_type = db_type

    def _connect(self) -> Any:  # Connection type can go here in place of `Any`
        # Do all of your connection stuff here...
        if self.db_type == "db_type_1":        
            conn = ...
            # ...
            # ...
            return conn
        elif self.db_type == "db_type_2":
            conn = ...
            # ...
            # ...
            return conn
        else:
            raise ...  # Raise some kind of error.


    def query(self, query: str) -> List[Tuple[str]]:
        """
        Run the query.  
        
        Note this connection is a mock, you'll have to replace 
        it with whatever connection object you're using.
        """
        conn = self._connect()
        if self.db_type == "db_type_1":
            return conn.run_a_query(query)
        elif self.db_type == "db_type_2":
            return conn.run_a_query_on_this_db(query)
        else:
            raise ... #  Raise some kind of error.
```

Notice two things.  

First, this is not a good looking hunk of code: there's going to be tons of if-else statements depending on how many DBs we need to add.  What if we forgot to add a DB to the "query" part?

Second, the connectors follow essentially the same "contract"; that is, they both have a method which _connects_ and which runs a _query_ and gets a result.  The difference is in how they're implemented...

### What's the contract?

Let's make some pseudo-code for how the contract looks for these two DBs.

```python
class DBConnector:
    """Connector for our DB."""

    # The init will be the same as above for both.
    def _connect(self) -> Any: ...
    def query(self, query: str) -> List[Tuple[str]]: ...
```

The idea is that we can use this as a parent class to both of the connectors, and have them "fill in" the methods with however that DB does its work.  **Abstract classes do this: they allow you to write up a "blueprint" for methods every child should implement.**

Let's look at how this looks in real python code.


```python
import abc  # Abstract Base Class library.
from typing import Any, List, Tuple


class DBConnector(abc.ABC):  # We must extend the ABC class for our Abstract Class.
    """Connector for our DB."""

    # The init is still the same.
    def __init__(self, db: Any, host: str, user: str, password: str) -> None:
        """Your amazing docstring goes here."""
        self.db = db
        self.host = host
        self.user = user
        self.password = password

    @abc.abstractmethod  # this makes the following function an abstract method!
    def _connect(self) -> Any:
        """Connect to the DB."""
        pass

    @abc.abstractmethod
    def query(self, query: str) -> List[Tuple[str]]:
        """Run the query."""
        pass
```

The ``abc.abstractmethod`` decorator tells Python that this is a method which a child class _must_ implement, or Python will throw an error.  This is good: it allows us to not miss anything when creating new connectors!

Let's make our two connectors using this abstract class DBConnector as a base class.


```python
class FirstDBConnector(DBConnector):
    """Connector for our db_type_1 DB."""

    def _connect(self) -> Any:  # We must define this now!
        """Connect to the DB."""
        conn = ...
        # do the connection logic here.
        return conn

    def query(self, query: str) -> List[Tuple[str]]:
        """Run the query."""
        conn = self._connect()
        return conn.run_a_query(query)


class SecondDBConnector(DBConnector):
    """Connector for our db_type_2 DB."""

    def _connect(self) -> Any:  # We must define this now!
        """Connect to the DB."""
        conn = ...
        # do some different connection logic here.
        return conn

    def query(self, query: str) -> List[Tuple[str]]:
        """Run the query."""
        conn = self._connect()
        return conn.run_a_query_on_this_db(query)
```

We now have two connectors that are following our abstract class contract.  What happens if we were to forget to define one of the methods in one of the classes?

Note that it's entirely possible to have "default" methods in the abstract class, so we can cut down on a lot of extra copy-pasting if these DB connectors shared a bunch of methods.  


```python
class ThirdDBConnector(DBConnector):
    """Connector for our db_type_2 DB."""

    def _connect(self) -> Any:  # We must define this now!
        """Connect to the DB."""
        conn = ...
        # do some different connection logic here.
        return conn


tdbconn = ThirdDBConnector("a", "b", "c", "d")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /home/james/repos/blog/_notebooks/abstract-classes-in-python.ipynb Cell 13' in <module>
          <a href='vscode-notebook-cell://wsl%2Bubuntu/home/james/repos/blog/_notebooks/abstract-classes-in-python.ipynb#ch0000029vscode-remote?line=6'>7</a>         # do some different connection logic here.
          <a href='vscode-notebook-cell://wsl%2Bubuntu/home/james/repos/blog/_notebooks/abstract-classes-in-python.ipynb#ch0000029vscode-remote?line=7'>8</a>         return conn
    ---> <a href='vscode-notebook-cell://wsl%2Bubuntu/home/james/repos/blog/_notebooks/abstract-classes-in-python.ipynb#ch0000029vscode-remote?line=9'>10</a> tdbconn = ThirdDBConnector("a", "b", "c", "d")


    TypeError: Can't instantiate abstract class ThirdDBConnector with abstract method query


We see that we get the following error:

`TypeError: Can't instantiate abstract class ThirdDBConnector with abstract method query`

That's exactly what we forgot to define! Python is preventing us from breaking the DBConnector contract. Nice.

### Some Other Notes

You can have default methods in your abstract class, and you can add methods unique to a  child class: it's no so strict that it gives the _exact_ form of the class, only what that class absolutely needs.

If you're using static or class methods, you can stack the decorators!

```python
@staticmethod
@abc.abstractmethod
def my_func(): ...
```

Check out some of the other stuff you can do in the [Python docs](https://docs.python.org/3/library/abc.html).

