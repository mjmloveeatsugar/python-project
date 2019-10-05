# -*- coding: utf-8 -*-
# @Author: DELL
# @Date:   2019-10-05 10:16:49
# @Last Modified by:   mjmloveeatsugar
# @Last Modified time: 2019-10-05 16:43:15
# Put this file in the dir '1.训练集'
import re
import os
import shutil
from collections import defaultdict


def categories_total(f_text):
    text_set = set()
    for item in f_text:
        id_ = pattern.findall(item)[0]
        text = pattern.sub('', item)
        text_set.add(text)
        print(len(text_set))


def classify(f_text):
    base_dir = 'lip_train\\lip_train'
    # ab_dir need to be changed
    ab_dir = r'E:\新网银行唇语识别竞赛数据\新网银行唇语识别竞赛数据\1.训练集\lip_train\lip_train'
    train_list = os.listdir(base_dir)
    new_dir = 'pic_classification'
    os.mkdir(new_dir)
    text_code = defaultdict(list)
    for item in f_text:
        id_ = pattern.findall(item)[0]
        text = pattern.sub('', item)
        text_code[text].append(id_)
    os.chdir(new_dir)
    for item, value in text_code.items():
        os.mkdir(item)
        for i in value:
            i = i[:-1]
            if i in train_list:
                shutil.copytree(os.path.join(ab_dir, i), os.path.join(item, i))
            else:
                print('no files in lip_train')


if __name__ == "__main__":
    pattern = re.compile(r'[0-9a-z\s]+')
    f_txt = open('lip_train.txt', 'r', encoding='utf-8').readlines()
    # categories_total(f_txt)
    '''
    Total 303 categories.
    Then we need to classify the pictures into 313 categories.
    '''
    classify(f_txt)
