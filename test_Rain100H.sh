#! bash 

# PReNet
python test_PReNet.py --logdir logs/Rain100H/PReNet6 --save_path results/Rain100H/PReNet --data_path datasets/test/Rain100H/rainy

# PRN
python test_PRN.py --logdir logs/Rain100H/PRN6 --save_path results/Rain100H/PRN6 --data_path datasets/test/Rain100H/rainy

# Proposed1
python test_Proposed1.py --logdir logs/Rain100H/Proposed1Adam --save_path results/Rain100H/Proposed1Adam --data_path datasets/test/Rain100H/rainy_image
python test_Proposed1.py --logdir logs/Rain100H/Proposed1SGD --save_path results/Rain100H/Proposed1SGD --data_path datasets/test/Rain100H/rainy_image
python test_Proposed1.py --logdir logs/Rain100H/Proposed1RMSProp --save_path results/Rain100H/Proposed1RMSProp --data_path datasets/test/Rain100H/rainy_image
python test_Proposed1.py --logdir logs/Rain100H/Proposed1CustomAdam --save_path results/Rain100H/Proposed1CustomAdam --data_path datasets/test/Rain100H/rainy_image