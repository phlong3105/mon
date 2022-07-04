#!/bin/bash

START_TIME="$(date -u +%s.%N)"

python3 aic22_retail_checkout.py --dataset "aic22retail" --config "testA_1.yaml"
python3 aic22_retail_checkout.py --dataset "aic22retail" --config "testA_2.yaml"
python3 aic22_retail_checkout.py --dataset "aic22retail" --config "testA_3.yaml"
python3 aic22_retail_checkout.py --dataset "aic22retail" --config "testA_4.yaml"
python3 aic22_retail_checkout.py --dataset "aic22retail" --config "testA_5.yaml"

#cd performance_test
#python3 combiner.py
#cd ..

END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
