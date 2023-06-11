#! bash

## Rain100L
python train_PReNet.py --preprocess True --save_path logs/Rain100L/PReNet --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20
python train_PRN.py --save_path logs/Rain100L/PRN --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20
python train_Proposed1.py --save_path logs/Rain100L/Proposed1Adam --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
python train_Proposed1.py --save_path logs/Rain100L/Proposed1SGD --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer SGD  
python train_Proposed1.py --save_path logs/Rain100L/Proposed1RMSProp --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer RMSProp  
python train_Proposed1.py --save_path logs/Rain100L/Proposed1CustomAdam --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 


## Rain100H
python train_PReNet.py --preprocess True --save_path logs/Rain100H/PReNet --data_path datasets/train/RainTrainH --batch_size 4 --epochs 25 --milestone 8 16 20
python train_PRN.py --save_path logs/Rain100H/PRN --data_path datasets/train/RainTrainH --batch_size 4 --epochs 25 --milestone 8 16 20
python train_Proposed1.py --save_path logs/Rain100H/Proposed1Adam --data_path datasets/train/RainTrainH --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
python train_Proposed1.py --save_path logs/Rain100H/Proposed1SGD --data_path datasets/train/RainTrainH --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer SGD  
python train_Proposed1.py --save_path logs/Rain100H/Proposed1RMSProp --data_path datasets/train/RainTrainH --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer RMSProp  
python train_Proposed1.py --save_path logs/Rain100H/Proposed1CustomAdam --data_path datasets/train/RainTrainH --batch_size 4 --epochs 25 --milestone 8 16 20 


## Rain12600
python train_PReNet.py --preprocess True --save_path logs/Rain1400/PReNet --data_path datasets/train/Rain12600 --batch_size 4 --epochs 4 --milestone 1 2 3
python train_PRN.py --save_path logs/Rain1400/PRN --data_path datasets/train/Rain12600 --batch_size 4 --epochs 4 --milestone 1 2 3
python train_Proposed1.py --save_path logs/Rain1400/Proposed1Adam --data_path datasets/train/Rain12600 --batch_size 4 --epochs 4 --milestone 1 2 3 --optimizer Adam
python train_Proposed1.py --save_path logs/Rain1400/Proposed1SGD --data_path datasets/train/Rain12600 --batch_size 4 --epochs 4 --milestone 1 2 3 --optimizer SGD  
python train_Proposed1.py --save_path logs/Rain1400/Proposed1RMSProp --data_path datasets/train/Rain12600 --batch_size 4 --epochs 4 --milestone 1 2 3 --optimizer RMSProp  
python train_Proposed1.py --save_path logs/Rain1400/Proposed1CustomAdam --data_path datasets/train/Rain12600 --batch_size 4 --epochs 4 --milestone 1 2 3 



