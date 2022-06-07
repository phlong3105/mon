#!/bin/bash

START_TIME="$(date -u +%s.%N)"

python3 main.py --dataset "aic21vehiclecounting" --config "cam_1.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_1_dawn.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_1_rain.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_2.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_2_rain.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_3.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_3_rain.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_4.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_4_dawn.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_4_rain.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_5.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_5_dawn.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_5_rain.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_6.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_6_snow.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_7.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_7_dawn.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_7_rain.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_8.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_9.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_10.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_11.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_12.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_13.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_14.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_15.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_16.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_17.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_18.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_19.yaml"
python3 main.py --dataset "aic21vehiclecounting" --config "cam_20.yaml"

#cd performance_test
#python3 combiner.py
#cd ..

END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
