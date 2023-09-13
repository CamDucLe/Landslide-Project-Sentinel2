## to setup ##
conda create --name landslide python==3.8.0
conda activate landslide
pip install -r requirements.txt

## to train ##
python3 main.py --is_train 1 --stored_folder 20-mono-f-i-23-RANet-multi-best --batch_size 12 --is_multi_res 1

## to test ## 
python3 main.py --is_train 0 --stored_folder 20-mono-f-i-23-RANet-multi-best --batch_size 12 --is_multi_res 1

# test data_generator ## 
python3 data_generator.py

## check value range of each band ##
python3 statistic.py

## test create model + see number of parameter, architecture ##
# uncomment architecture to be trained #
python3 model.py

## test data generator ##
python3 data_generator.py