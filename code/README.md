## to train ##
python3 main.py --is_train 1 --stored_folder 20-mono-f-i-21 --batch_size 1 --is_multi_res 0

## to test ## 
python3 main.py --is_train 0 --stored_folder 20-mono-f-i-21 --batch_size 1 --is_multi_res 0

## check value range of each band ##
python3 statistic.py

## test create model + see number of parameter, architecture ##
python3 model.py

## test data generator ##
python3 data_generator.py
