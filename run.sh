python codem/utils/kfold.py
python codem/train/train.py --num_train_epochs 10  --model_type nezha --model_path nezha_base --save_path nezha_base --pretrain
python codem/train/train.py --num_train_epochs 10  --model_type nezha --model_path nezha_wwm --save_path nezha_wwm --pretrain 
python codem/train/train.py --num_train_epochs 15  --model_type bert --model_path mac_bert --save_path mac_bert --pretrain 

python codem/train/train.py --num_train_epochs 3 --fgm  --model_type bert --model_path mac_bert --save_path mac_bert --logging_steps 445
python codem/train/train.py --num_train_epochs 3 --fgm  --model_type nezha --model_path nezha_base --save_path nezha_base  --logging_steps 445
python codem/train/train.py --num_train_epochs 3 --fgm  --model_type nezha --model_path nezha_wwm --save_path nezha_wwm   --logging_steps 445
python codem/inference/test.py --do_predict 


