import pickle
import os
import glob

def readPKL():
    files = glob.glob('../predictions_folder/*.pkl')
    for file_name in files:
        with open(file_name, 'rb') as f:
            x = pickle.load(f)
         
        if len(x) > 0:
            print(x[0])
            return
        else:
            print(x)

if __name__ == "__main__":
    readPKL()
