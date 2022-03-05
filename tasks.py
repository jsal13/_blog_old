"""Tasks file for `invoke`.

Notes:  Type-Hinting is not able to be used until
https://github.com/pyinvoke/invoke/issues/357 is closed.
"""

import datetime
import os

from invoke import task


@task
def nb2md(context, nb_name=None):  # type: ignore
    """
    Take notebook and make post md.

    `nb_name` is the name of the notebook with no suffix.
    """
    dt = str(datetime.date.today())
    cmd = (
        f"jupyter nbconvert --to markdown "
        f"./_notebooks/{nb_name}.ipynb "  # type: ignore
        f"--output ../_posts/{dt}-{nb_name}.md"
    )
    print(cmd)
    context.run(cmd)


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
