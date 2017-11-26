#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random

TEST_PER=0.1 #10%

def main():

    caltech_101_dir = "/home/gachiemchiep/workspace/DL/data/101_ObjectCategories"

    data_dirs = [os.path.join(caltech_101_dir, data_dir) for data_dir in os.listdir(caltech_101_dir)
                 if os.path.isdir(os.path.join(caltech_101_dir, data_dir))]

    data_dirs = sorted(data_dirs)

    train_fid = open("Caltech101_train.txt", "w")
    test_fid = open("Caltech101_test.txt", "w")

    for idx, data_dir in enumerate(data_dirs):

        dir_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)
                     if os.path.isfile(os.path.join(data_dir, file))]

        random.shuffle(dir_files)

        train_count = int(len(dir_files) * (1 - TEST_PER))

        train_files = dir_files[0: train_count]
        test_files = dir_files[train_count: ]

        for idy, dir_file in enumerate(dir_files):
            line = "%s %s\n" % (dir_file, idx)
            if idx > train_count:
                test_fid.write(line)
            elif idx < train_count:
                train_fid.write(line)

    train_fid.close()
    test_fid.close()

    caltech_256_dir = "/home/gachiemchiep/workspace/DL/data/256_ObjectCategories"

    data_dirs = [os.path.join(caltech_256_dir, data_dir) for data_dir in os.listdir(caltech_256_dir)
                 if os.path.isdir(os.path.join(caltech_256_dir, data_dir))]

    data_dirs = sorted(data_dirs)

    train_fid = open("Caltech256_train.txt", "w")
    test_fid = open("Caltech256_test.txt", "w")

    for idx, data_dir in enumerate(data_dirs):

        dir_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)
                     if os.path.isfile(os.path.join(data_dir, file))]

        random.shuffle(dir_files)

        train_count = int(len(dir_files) * (1 - TEST_PER))

        train_files = dir_files[0: train_count]
        test_files = dir_files[train_count:]

        for idy, dir_file in enumerate(dir_files):
            line = "%s %s\n" % (dir_file, idx)
            if idx > train_count:
                test_fid.write(line)
            elif idx < train_count:
                train_fid.write(line)

    train_fid.close()
    test_fid.close()



if __name__ == '__main__':
    main()