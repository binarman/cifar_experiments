#!/bin/bash

file -E logs %> /dev/null || mkdir logs
python3 "./${1}.py" | tee "logs/${1}"
