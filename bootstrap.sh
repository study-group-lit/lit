#!/bin/bash
# Setup Python environment
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt