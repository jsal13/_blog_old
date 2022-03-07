"""Tasks file for `invoke`.

Notes:  Type-Hinting is not able to be used until
https://github.com/pyinvoke/invoke/issues/357 is closed.
"""

import os
import re

from invoke import task


@task
def nb2md(context, nb_name=None):  # type: ignore
    """
    Take notebook and make post md.

    `nb_name` is the name of the notebook with no suffix.
    """
    # Open the original file to get the date of posting.
    nb_path: str = f"./_notebooks/{nb_name}.ipynb"

    with open(nb_path, "r", encoding="utf-8") as f:
        f_raw_nb = f.read()

    date_pattern = r"date:.*?(\d{4}-\d{2}-\d{2})"
    dt = re.findall(date_pattern, f_raw_nb)[0]

    md_path: str = f"./_posts/{dt}-{nb_name}.md"

    # Create and run command to convert nb to markdown.
    cmd = f"jupyter nbconvert --to markdown {nb_path} " f"--output ../{md_path}"
    print(cmd)
    context.run(cmd)

    # Convert assets to site-ready assets.
    with open(md_path, "r", encoding="utf-8") as f:
        f_raw_md = f.read()

    nb_asset_pattern = r"!\[\]\(../(assets/images/.*?)\)"
    md_asset_pattern = r"!\[\]\({{site.baseurl}}/\1\)"
    print(f"All asset replacements: {re.findall(nb_asset_pattern, f_raw_md)}")
    f_raw_md = re.sub(nb_asset_pattern, md_asset_pattern, f_raw_md)

    # Write results back to md file.
    with open(md_path, "w+", encoding="utf-8") as f:
        f.write(f_raw_md)


@task
def docs(context):  # type: ignore
    """Generate Sphinx documentation."""
    if os.name == "nt":  # if windows...
        context.run(".\\docs\\make.bat html")
    else:
        context.run("cd docs && make html && cd ..")


@task
def test(context):  # type: ignore
    """Run Pytest."""
    context.run("pytest --cov=stuff tests/")


@task
def install_precommit(context):  # type: ignore
    """Install pre-commit into githooks."""
    context.run("pre-commit install")


@task
def tox(context):  # type: ignore
    """Run Tox."""
    context.run("tox")


@task(pre=[install_precommit])
def init(_context):  # type: ignore
    """Init the repository.

    - Install pre-commit hooks.
    """
    print("Installing pre-commit...")
