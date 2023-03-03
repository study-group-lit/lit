#!/bin/env sh
# Setup Python environment
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt