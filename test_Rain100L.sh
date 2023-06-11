#! bash 

# PReNet
python test_PReNet.py --logdir logs/Rain100L/PReNet --save_path results/Rain100L/PReNet --data_path datasets/test/Rain100L/rainy

# PRN
python test_PRN.py --logdir logs/Rain100L/PRN --save_path results/Rain100L/PRN --data_path datasets/test/Rain100L/rainy

# Proposed1
python test_Proposed1.py --logdir logs/Rain100L/Proposed1Adam --save_path results/Rain100L/Proposed1Adam --data_path datasets/test/Rain100L/rainy_image
python test_Proposed1.py --logdir logs/Rain100L/Proposed1SGD --save_path results/Rain100L/Proposed1SGD --data_path datasets/test/Rain100L/rainy_image
python test_Proposed1.py --logdir logs/Rain100L/Proposed1RMSProp --save_path results/Rain100L/Proposed1RMSProp --data_path datasets/test/Rain100L/rainy_image
python test_Proposed1.py --logdir logs/Rain100L/Proposed1CustomAdam --save_path results/Rain100L/Proposed1CustomAdam --data_path datasets/test/Rain100L/rainy_image