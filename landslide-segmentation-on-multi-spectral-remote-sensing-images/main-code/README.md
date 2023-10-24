
## prepare folders like description below ##
- project folder</br>
        + code (you are here):</br>
        + results: create "stored_folder" related to different experiments, like</br>
                ~ 01_UNET_MULTI_RESOLUTION
                .......
        + predictions_folder:</br>
        + dataset (don't create this one):  when you run main, it will automatically create this folder and the following folders:</br>
                ~ train/test/val folders: each of them will have img/mask folder</br>
        + zip_data : place Landslide4Sense .zip file here


## to setup environemnt ##
conda create --name landslide python==3.8.0
conda activate landslide
pip install -r requirements.txt


## to train ##
python3 main.py --is_train 1 --stored_folder <your_folder_name> --batch_size <batch_size> --is_multi_res 1

=> example: python3 main.py --is_train 1 --stored_folder 01_RAUNET_MULTI_RESOLUTION  --batch_size 12 --is_multi_res 1


## to test ## 
python3 main.py --is_train 0 --stored_folder <your_folder_name> --batch_size <batch_size> --is_multi_res 1


## use HPC server ##
=> to train/test:  sbatch run.sh
=> to make prediction large remote sensing images: sbatch predict.sh


## to do k-fold cross validation, change test_folder number in "main.py" ##

## to apply thresholding in testing change is_post_processing to 0 , 1 , 2 , 3 , 4 ##

## to try out U-Net multi-resolution, traditional U-Net or other models -> just take them out of folder-models and change is_multi_res to 0 or 1 ##


# test data_generator ## 
python3 data_generator.py

## check value range of each band ##
python3 statistic.py

## test create model + see number of parameter, architecture ##
python3 model.py
