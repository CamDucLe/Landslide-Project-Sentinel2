import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import psutil
import random
from datetime import datetime
from io import BytesIO



def loadSmallChunk(batch_size=27, chunk_num=1):
    parquet_file = pq.ParquetFile('/home/pielorzj/storage/gaia/dataframe/carinthia_10m.parquet')
    ite = 0
    for i in parquet_file.iter_batches(batch_size=batch_size):
        small_chunk = i.to_pandas()
        ite += 1
        if ite == chunk_num:
            break
    del parquet_file, i, ite

    return small_chunk


def getColumnsName(data_chunk=None):
    if data_chunk is not None:
        return data_chunk.columns.to_list()
    else:
        parquet_file = pq.ParquetFile('/home/pielorzj/storage/gaia/dataframe/carinthia_10m.parquet')
        for i in parquet_file.iter_batches(batch_size=10):
            small_chunk = i.to_pandas()
            break

        del parquet_file, i

        return small_chunk.columns.to_list()


def loadWholeDataset(cols_name, by_col=False):
    if by_col == True:
        cols_name = [cols_name]

    print('RAM % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


    ### read .parquet file
    parquet_file = '/home/pielorzj/storage/gaia/dataframe/carinthia_10m.parquet'
    #df = pd.read_parquet(parquet_file, columns=cols_name, engine='pyarrow')
    df = pq.read_table(parquet_file, columns=cols_name).to_pandas()
    
    ### read .arrow file
    #arrow_file = '/home/pielorzj/storage/gaia/dataframe/carinthia_10m.arrow'
    #df = pa.ipc.open_file(arrow_file).read_all()
    #df = df.to_pandas()

    print('RAM % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    del parquet_file
    
    return df


def readChunkPartion(file_name, is_train=True, train_ratio=0.75, is_pos=True):
    ## read .arrow file
    part_chunk = pa.ipc.open_file(file_name).read_all()
    part_chunk = part_chunk.to_pandas() 

    ## neu la train thi 75% pos va 50% neg -> test thi 25% pos va 50% neg
    if is_pos==False:
        train_ratio = 0.5

    if is_train:
        part_chunk = part_chunk[:int(part_chunk.shape[0]*train_ratio)]
    else: 
        part_chunk = part_chunk[int(part_chunk.shape[0]*train_ratio):]
    
    return part_chunk


def loadChunks(is_train=True):
    pos_file_name = '/home/phaml/storage/01_project/04_AI_for_Earth/05_landslide_dataset/chunks/pos/carinthia_slides.arrow'
    neg_dir = '/home/phaml/storage/01_project/04_AI_for_Earth/05_landslide_dataset/chunks/neg/'
    
    ## load each chunk into a list of chunks
    data_chunks  = []
    data_chunks.append(readChunkPartion(pos_file_name, is_train, is_pos=True))
    for i in range(1,10):
        neg_file_name = neg_dir + 'partition=' + str(i) + '/' + 'part-0.arrow'
        data_chunks.append(readChunkPartion(neg_file_name, is_train, is_pos=False))
    
    ## turn into pandas data frame and shuffle
    df = pd.concat(data_chunks)
    df = df.sample(frac = 1)
    df = df.dropna(axis=0)
    del data_chunks

    return df
    

if __name__ == "__main__":
    #chunk = loadSmallChunk()
    #print(chunk.head())
    #cols_name = getColumnsName(chunk)
    #print(getColumnsName(chunk))

    #print('\n\n===================================\n\n')
    
    #df = loadWholeDataset(cols_name)
    #print(df.head(3), df.shape)
    
    #del chunk, df

    print('RAM % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    df = loadChunks(True)
    print('\nTrain df: ', df.shape)
    df = loadChunks(False)
    print('\nTest df: ',df.shape)

    print('RAM % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    del df
