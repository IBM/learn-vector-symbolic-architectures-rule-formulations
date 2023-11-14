#!/bin/bash

# 
EPOCHS=50 # change between 1 and 50 epochs
SELECTOR="sample" # select between 'sample' and 'weight'
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config center_single 
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config distribute_four
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config distribute_nine
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config left_right
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config up_down
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config in_out_single
python3 src/rule_level_backend.py  --epochs $EPOCHS --vsa_conversion --vsa_selection --rule_selector $SELECTOR --shared_rules --config in_out_four