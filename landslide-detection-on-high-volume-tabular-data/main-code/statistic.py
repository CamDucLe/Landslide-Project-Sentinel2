import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pandas as pd
import numpy as np
import os
from datetime import datetime

from load_data import *


def getBasicInfo():
    cols_name = getColumnsName(data_chunk=None)
    print("Number of Features (columns) in dataframe: ", len(cols_name))
    null_list = []
    max_percent_null = 0
    max_feature_null = "Nonee"
    for i in range(len(cols_name)):
        print('\n\n =====  Feature ', i, ": ", cols_name[i], ' =====')
        feature_i = loadWholeDataset(cols_name[i], by_col=True)
        
        print("Data type: ", feature_i.dtypes[0])
        print("Number of records: ", feature_i.shape[0])

        #if feature_i.to_numpy().dtype == np.float64():
        print(feature_i.describe())

        print("Number of rows has missing value-Null: ", feature_i.isna().sum()[0] )
        percent_null = feature_i.isna().sum()[0] / feature_i.shape[0] * 100
        print("Percentage of Null / total record: ", percent_null, " %")
        
        if percent_null > max_percent_null:
            max_percent_null = percent_null
            max_feature_null = cols_name[i]
        
        if feature_i.isna().sum()[0] != 0:
            null_list.append( (cols_name[i], round(percent_null,3)) )
        #break

    print("\n Null list: ", null_list)
    print("\n Feature with maximum % Null: ", max_feature_null, "  -- ", max_percent_null, " %")

    del feature_i, cols_name, max_percent_null, max_feature_null, null_list

    return 


def drawBoxPlot():
    cols_name = getColumnsName(data_chunk=None)
    print("Number of Features (columns) in dataframe: ", len(cols_name))

    for i in range(len(cols_name)):
        print('\n\n =====  Feature ', i, ": ", cols_name[i], ' =====')
        feature_i = loadWholeDataset(cols_name[i], by_col=True)
        
        if feature_i.to_numpy().dtype == np.float64():
            figure_box = feature_i.plot.box()
            figure_box = figure_box.get_figure()
            fig_name_box = 'plot_figure/box_' + str(i) + '_' + cols_name[i] + '.png'
            figure_box.savefig(fig_name_box)
        elif feature_i.dtypes[0] == 'category':
            figure_bar = feature_i.value_counts().plot.bar()
            figure_bar = figure_bar.get_figure()
            fig_name_bar = 'plot_figure/bar_' + str(i) + '_' + cols_name[i] + '.png'
            figure_bar.savefig(fig_name_bar)
        
        print('RAM % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    
    del feature_i, cols_name

    return


def drawKdePlot():
    cols_name = getColumnsName(data_chunk=None)
    print("Number of Features (columns) in dataframe: ", len(cols_name))

    for i in range(len(cols_name)):
        print('\n\n =====  Feature ', i, ": ", cols_name[i], ' =====')
        feature_i = loadWholeDataset(cols_name[i], by_col=True)
        if i != 28: 
            if feature_i.to_numpy().dtype == np.float64():
                figure_kde = feature_i.plot.kde()
                figure_kde = figure_kde.get_figure()
                fig_name_kde = 'plot_figure/kde_' + str(i) + '_' + cols_name[i] + '.png'
                figure_kde.savefig(fig_name_kde)
            
                del figure_kde, fig_name_kde
        
        print('RAM % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    del feature_i, cols_name

    return


def drawBarPlot():
    cols_name = getColumnsName(data_chunk=None)
    print("Number of Features (columns) in dataframe: ", len(cols_name))

    for i in range(len(cols_name)):
        print('\n\n =====  Feature ', i, ": ", cols_name[i], ' =====')
        feature_i = loadWholeDataset(cols_name[i], by_col=True)

        if feature_i.dtypes[0] == 'category':
            figure_bar = feature_i.value_counts().plot.bar()
            figure_bar = figure_bar.get_figure()
            fig_name_bar = 'plot_figure/bar_' + str(i) + '_' + cols_name[i] + '.png'
            figure_bar.savefig(fig_name_bar)
            
            del figure_bar

        print('RAM % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    del feature_i, cols_name

    return


if __name__ == "__main__":
    getBasicInfo()
    #drawBoxPlot()
    #drawBarPlot()
    #drawKdePlot()












