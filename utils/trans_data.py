import json
import csv
import sys
import os
import datetime
import numpy as np
from scipy import stats
import pandas as pd


def feature_eng(in_file, input_T):
    sale_T = 14
    data_list = {}
    f = csv.reader(open(in_file, 'r'))
    for line in f:
        now_date = datetime.date(int(line[0][0:4]), int(line[0][4:6]), int(line[0][6:8]))
        if line[1] not in data_list.keys():
            whSapName = line[2]
            p = {}
            p['id'] = line[1]
            p['goodsThirdName'] = line[3]
            p['goodsClassifyName'] = line[4]
            p['whSapCode'] = line[2]
            p['whSapName'] = line[2]
            p['sale_time_series'] = {}
            p['uv_time_series'] = {}
            p['cart_time_series'] = {}
            p['like_time_series'] = {}
            data_list[line[1]] = p
        if now_date not in data_list[line[1]]['sale_time_series'].keys():
            data_list[line[1]]['sale_time_series'][now_date] = int(line[18])
            data_list[line[1]]['uv_time_series'][now_date] = int(line[8])
            data_list[line[1]]['cart_time_series'][now_date] = int(line[10])
            data_list[line[1]]['like_time_series'][now_date] = int(line[11])
        else:
            data_list[line[1]]['sale_time_series'][now_date] += int(line[18])
            data_list[line[1]]['uv_time_series'][now_date] += int(line[8])
            data_list[line[1]]['cart_time_series'][now_date] += int(line[10])
            data_list[line[1]]['like_time_series'][now_date] += int(line[11])


    e = datetime.date(2015, 11, 30)
    s = datetime.date(2014, 10, 10)

    begin = e - datetime.timedelta(days=sale_T)
    end = e
    watch = 0
    data_dict = []
    while (begin.__sub__(s)) >= datetime.timedelta(days=sale_T):
        all_sale, all_feature, all_id, all_label = get_all_sale(begin, end, data_list , input_T, sale_T)
        sale_dict = {}
        sale_dict['watch'] = watch
        sale_dict['whSapName'] = whSapName
        sale_dict['input_end_date'] = datetime.datetime.strftime(end, "%Y-%m-%d")
        sale_dict['id'] = all_id
        sale_dict['X'] = all_sale.tolist()
        sale_dict['F'] = all_feature.tolist()
        sale_dict['y'] = all_label
        data_dict.append(sale_dict)

        # update watch
        '''
        if watch==0:
            begin = begin - datetime.timedelta(days=sale_T)
            end = end - datetime.timedelta(days=sale_T)
        else:
            #begin = begin - datetime.timedelta(days=1)
            #end = end - datetime.timedelta(days=1)
            begin = begin - datetime.timedelta(days=sale_T)
            end = end - datetime.timedelta(days=sale_T)
        '''
        begin = begin - datetime.timedelta(days=sale_T)
        end = end - datetime.timedelta(days=sale_T)
        watch += 1
    
    return data_dict

def get_all_sale(begin, end, data_list, input_T, sale_T):
    key_list = list(data_list.keys())
    all_sale = np.zeros((input_T, len(key_list)))
    all_feature = np.zeros((input_T, len(key_list), 3))
    all_label = []
    for i, k in enumerate(key_list):
        data = data_list[k]
        for j in range(input_T):
            day = end - datetime.timedelta(days=input_T) + datetime.timedelta(days=j)
            if day in data['sale_time_series'].keys():
                all_sale[j, i] = data['sale_time_series'][day]
                all_feature[j, i, 0] = data['uv_time_series'][day]
                all_feature[j, i, 1] = data['cart_time_series'][day]
                all_feature[j, i, 2] = data['like_time_series'][day]
        label = 0
        for j in range(sale_T):
            day = end + datetime.timedelta(days=j)
            if day in data['sale_time_series'].keys():
                label += data['sale_time_series'][day]
        all_label.append(label)
    return all_sale, all_feature, key_list, all_label
