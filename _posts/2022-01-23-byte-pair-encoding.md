---
title:  "Byte Pair Encoding: A Byte-Sized Introduction"
date:   2022-01-23

description: Introduction to the Byte Pair Encoding algorithm.
categories: python algorithms compression

excerpt: "While going over some Natural Language Processing topics, I stumbled on the _byte pair encoding_ algorithm.  I thought that, given its usefulness in a number of NLP applications, that it'd be fairly complex and quite difficult to understand."

classes: wide

header:
  overlay_filter: rgba(0, 146, 202, 0.8)
  overlay_image: /assets/images/title_compression_spring.jpg
  caption: "Photo Credit: [**Jean-Jacques Milan**](https://commons.wikimedia.org/wiki/File:Ressort_de_compression.jpg#metadata)"
---
## Introduction

While going over some Natural Language Processing topics, I stumbled on the _byte pair encoding_ algorithm.  I thought that, given its usefulness in a number of NLP applications, that it'd be fairly complex and quite difficult to understand &mdash; but it turns out that it's pretty straight-forward!  Let's look into it a bit and give some examples.  First, our imports.


```python
import string
from collections import Counter
import random
```

## The Problem

**Problem:** We have a string (for now, let's say it's of lowercase letters) and we'd like to compress it; that is, we'd like to make the string shorter while containing the same information as before.

Let's detail one potential solution.


```python
random.seed(12345)  # Set seed for reproducability.
data = "".join(random.choices("abcd", weights=[10, 5, 2, 1], k=20))
data
```




    'aabaaabaaabaaadadaaa'



Let's split this up into pairs of letters.  Then, we'll count to see how many times each pair comes up.


```python
random.seed(12345)  # Set seed for reproducability.
data = "".join(random.choices("abcd", weights=[10, 5, 2, 1], k=20))

# Split data up into pairs.  Each is a byte, so these are _Byte Pairs_.
byte_pairs = [f"{data[idx]}{data[idx + 1]}" for idx in range(len(data) - 1)]
byte_pair_counts = Counter(byte_pairs).most_common()

print(byte_pair_counts)
```

    [('aa', 9), ('ab', 3), ('ba', 3), ('ad', 2), ('da', 2)]


(_Note: when we do this, that we could also ignore the first letter to get an entirely different pairing.  It may be useful to try this to optimize the compression but, for now, let's stick with this._)

We see that ``aa`` comes up quite a bit, followed by ``ba``and ``ab``.  Let's take the most common byte pair (``aa``) and replace it with a single letter that doesn't appear anywhere else in the data; for simplicity, let's use capital letters (Z, Y, X, ...) for replacement.


```python
# Make two functions here for reusability.
def get_byte_pairs_from_data(data: str) -> list[tuple[str, int]]:
    """Get byte_pairs from most common to least common from ``data``."""
    byte_pairs = [f"{data[idx]}{data[idx + 1]}" for idx in range(len(data) - 1)]
    return Counter(byte_pairs).most_common()


def replace_byte_pair(data: str, pair: str, replacement: str) -> str:
    """Replace instances of ``pair``in ``data``in a serial manner."""
    while data.count(pair) > 0:
        data = data.replace(pair, replacement, 1)  # Replace the first occurance.
    return data


# Now let's try them out!
random.seed(12345)  # Set seed for reproducability.
data = "".join(random.choices("abcd", weights=[10, 5, 2, 1], k=20))
byte_pair_counts = get_byte_pairs_from_data(data)

# Make our replacement mapping / table.
replacement_mapping = [
    (byte_pair_counts[i][0], string.ascii_uppercase[::-1][i])
    for i in range(len(byte_pair_counts))
]
print(replacement_mapping)
```

    [('aa', 'Z'), ('ab', 'Y'), ('ba', 'X'), ('ad', 'W'), ('da', 'V')]



```python
# What happens when we replace one value?
replace_byte_pair(data, "aa", "Z")
```




    'ZbZabZabZadadZa'




```python
# Let's replace all the values now!
compressed_data = data[::]  # Copy data.
for pair in replacement_mapping:
    compressed_data = replace_byte_pair(compressed_data, pair[0], pair[1])

print(compressed_data)

print()
print("No Compression: " + "".join(byte_pairs))
print("Compression:    " + "".join(compressed_data))
print()
print(f"Original Length: {len(data)}\nCompressed Length: {len(compressed_data)}")
```

    ZbZYZYZWWZa

    No Compression: aaabbaaaaaabbaaaaaabbaaaaaaddaaddaaaaa
    Compression:    ZbZYZYZWWZa

    Original Length: 20
    Compressed Length: 11


That's pretty good!  We can make this a bit larger if we'd like to see how good this compression works...


```python
random.seed(12345)  # Set seed for reproducability.
data = "".join(random.choices("abcd", weights=[10, 5, 2, 1], k=20_000))
byte_pair_counts = get_byte_pairs_from_data(data)

replacement_mapping = [
    (byte_pair_counts[i][0], string.ascii_uppercase[::-1][i])
    for i in range(len(byte_pair_counts))
]

compressed_data = data[::]  # Copy data.
for pair in replacement_mapping:
    compressed_data = replace_byte_pair(compressed_data, pair[0], pair[1])

print(f"Original Length: {len(data)}\nCompressed Length: {len(compressed_data)}")
```

    Original Length: 20000
    Compressed Length: 11281


At some point, we'll need to decode this.  We can do this by looking at our replacement mapping and replacing in _the reverse order_ (first in, last out).


```python
decompressed_data = compressed_data[::] # Copy data.
for pair in replacement_mapping[::-1]:
    decompressed_data = replace_byte_pair(decompressed_data, pair[1], pair[0])
```


```python
# Got the same data back!
print(data == decompressed_data)
```

    True


## Things To Notice

First, there is a lot of room for optimization in the algorithm above: I've attempted to make the algorithm a bit more readable at the cost of optimization.  Optimization of the above is left as an exercise to the reader.

Second, we don't gain a whole lot by replacing pairs that don't occur frequently (especially those which occur once), so it's possible to remove them.  It's also possible to do this process recursively on an encoded set of data to get a bit more compression.  For example, if we encoded a string to ``ZaYaZaZZ`` we might see that ``Za`` occurs a few times and want to compress this down further to, say, ``XYaXZZ``.  This may not seem significant in this case, but it may save a lot of space for extremely large, repetitive files.

## Can we do this with anything besides strings of letters?

Sure.  One common thing to do in Natural Language Processing is to pair adjacent words into "bigrams": for example, "Mary Had A Little Lamb" goes to ``[("Mary", "Had"), ("Had", "A"), ("A", "Little"), ("Little", "Lamb")]``.  From here, we can see how we might apply the above techniques.  For this sentence there is no better encoding, but one can imagine a long novel or logfiles with a significant amount of text repetition where this would compress the original data quite a bit.
