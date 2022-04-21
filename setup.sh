#!/bin/bash
/home/ubuntu/miniconda3/envs/torch/bin/python -m build
/home/ubuntu/miniconda3/envs/torch/bin/python -m pip install dist/*.tar.gz
