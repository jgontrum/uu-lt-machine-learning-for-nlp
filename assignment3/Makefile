.PHONY: clean jupyter jupyter-headless

TAG=$(shell git symbolic-ref -q --short HEAD)

# Set the environment variable `PYTHON3` to specify the Python binary used.
PYTHON3?=python3.4

PORT=8899

all: env/bin/python

env/bin/python:
	$(PYTHON3) -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install wheel
	env/bin/pip install -r requirements.txt
	env/bin/jupyter nbextensions_configurator enable --user
	env/bin/jupyter tensorboard enable --user

jupyter:
	env/bin/jupyter notebook

jupyter-headless:
	env/bin/jupyter notebook --no-browser --port=$(PORT)

clean:
	rm -rfv bin develop-eggs dist downloads eggs env parts .cache build
	rm -fv .DS_Store .coverage .installed.cfg bootstrap.py .coverage
	find . -name '*.pyc' -exec rm -fv {} \;
	find . -name '*.pyo' -exec rm -fv {} \;
	find . -depth -name '*.egg-info' -exec rm -rfv {} \;
	find . -depth -name '__pycache__' -exec rm -rfv {} \;\
