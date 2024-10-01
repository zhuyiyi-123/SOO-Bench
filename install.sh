#!/usr/bin/bash
# install revive
pip install -e ./revive_hybrid

# install soo-bench
pip install -e .

# install dependencies of algorithms
pip install -r requirements.txt
