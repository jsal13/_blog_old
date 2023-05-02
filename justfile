set shell := ["zsh", "-cu"]
set positional-arguments

default:
    just --list

install:
    bundle install

serve: install
    bundle exec jekyll serve

@lint notebook:
    echo "> Formating ${{notebook}}..."
    black -q {{notebook}}

@convert notebook:
    echo "> Converting ${{notebook}}..."
    python ./convert.py {{notebook}}

@lint-and-convert notebook:
    just lint {{notebook}}
    just convert {{notebook}}
    echo ""  # Print a blank line to separate logs.

@lint-and-convert-all:
    for f in ./_notebooks/*.ipynb; do \
      just lint-and-convert "$f" ; \
    done