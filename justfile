set shell := ["zsh", "-cu"]
set positional-arguments

default:
    just --list

nb2md notebook:
    python ./convert.py {{notebook}}
    
serve:
    bundle exec jekyll serve