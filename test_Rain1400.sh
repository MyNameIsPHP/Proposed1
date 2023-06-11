#! bash

# PReNet
python test_PReNet.py --logdir logs/Rain1400/PReNet --save_path results/Rain1400/PReNet --data_path datasets/test/Rain1400/rainy_image

# PRN
python test_PRN.py --logdir logs/Rain1400/PRN --save_path results/Rain1400/PRN --data_path datasets/test/Rain1400/rainy_image

# Proposed1
python test_Proposed1.py --logdir logs/Rain1400/Proposed1Adam --save_path results/Rain1400/Proposed1Adam --data_path datasets/test/Rain1400/rainy_image
python test_Proposed1.py --logdir logs/Rain1400/Proposed1SGD --save_path results/Rain1400/Proposed1SGD --data_path datasets/test/Rain1400/rainy_image
python test_Proposed1.py --logdir logs/Rain1400/Proposed1RMSProp --save_path results/Rain1400/Proposed1RMSProp --data_path datasets/test/Rain1400/rainy_image
python test_Proposed1.py --logdir logs/Rain1400/Proposed1CustomAdam --save_path results/Rain1400/Proposed1CustomAdam --data_path datasets/test/Rain1400/rainy_image
