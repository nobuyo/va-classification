#!/bin/bash

# python3 vac_train.py

for i in $(seq 3 55); do
  python3 vac_train.py --model vac$(expr $i - 1).model --division $i
  # echo "python3 vac_train.py --model vac$(expr $i - 1).model --division $i"
done