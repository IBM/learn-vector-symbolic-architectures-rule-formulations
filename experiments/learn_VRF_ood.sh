#!/bin/bash

# 
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Type'  --gen_rule 'Constant'
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Type'  --gen_rule 'Progression'
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Type'  --gen_rule 'Distribute_Three' 
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Size'  --gen_rule 'Constant'
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Size'  --gen_rule 'Progression' 
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Size'  --gen_rule 'Distribute_Three' 
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Size'  --gen_rule 'Arithmetic' 
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Color' --gen_rule 'Constant'
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Color' --gen_rule 'Progression'
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Color' --gen_rule 'Distribute_Three' 
python3 src/rule_level_backend.py  --vsa_conversion --vsa_selection --shared_rules --config center_single  --gen_attribute 'Color' --gen_rule 'Arithmetic' 