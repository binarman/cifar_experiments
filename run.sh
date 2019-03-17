#!/bin/bash

python3 "./${1}.py" | tee "logs/${1}"
