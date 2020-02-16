#!/bin/bash

file -E logs > /dev/null || mkdir logs
for i in "$@"; do
  python3 "./${i}.py" | tee "logs/${i}"
done
