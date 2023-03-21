# Blog.
My personal blog.  Math & Data & Python & Sundary.

# Prereqs
Install `poetry` and `just`.
- `pip install poetry`
- [Just](https://github.com/casey/just#installation)

Configure Poetry to use in-project virtual environments:

```shell
poetry config virtualenvs.in-project true
```

Make sure that you initialize the repository with 

```shell
poetry install
```

# Creating a Post


0. Use `poetry shell` to get into a poetry venv.
1. Create a Jupyter notebook with the entire blog post.
2. In the Jupyter notebook, create a preamble similar to this (see other notebooks for other examples):

    ```yaml
    title:  "My Cool Title"
    date:   2023-03-21

    description: A cool post about something.
    categories: python, cool

    excerpt: "Maybe the first sentence..."

    classes: wide

    header:
    overlay_filter: rgba(0, 146, 202, 0.8)
    overlay_image: /assets/images/my_cool_picture.jpg
    caption: "Photo Credit: Who?"
    ---
    ```

3. Convert the Notebook to a MD file for Jekyll consumption:

    ```shell
    just nb2md ./_notebooks/abstract-classes-in-python.ipynb
    ```

4. Check the post in `./posts/` for accuracy.

# Running Locally

Before pushing your posts, it may be desirable to see if they work locally.

Follow the steps detailed [here](https://jekyllrb.com/docs/) to set up and bring up the website locally.