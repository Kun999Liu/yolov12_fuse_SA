# -*- coding: utf-8 -*-
# @Time    : 2025/8/30 16:19
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : tif.py
# @Software: PyCharm

"""
Describe:
"""
from __future__ import division
import sys
import numpy as np
from osgeo import gdal


def readTif(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    # projection = dataset.GetProjection()  # 投影
    # geotrans = dataset.GetGeoTransform()  # 几何信息
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    img_array = dataset.ReadAsArray(0, 0, im_width, im_height)
    # img_array = img_array[:3, :, :]  # 取R G B三个波段
    # img_array = img_array[1:4, :, :]  # 取G B NIR三个波段
    # # img_array = img_array[[0,1,3], :, :]  # 取R G NIR三个波段
    # img_array = img_array[[0, 2, 3], :, :] # 取b r NIR三个波段
    '''校正后处理'''
    imgScale = img_array / 10000
    img = imgScale * 255
    img1 = np.round(img).astype(np.uint8)
    img2 = np.transpose(img1, (1, 2, 0))
    # return im_width, im_height, im_bands, projection, geotrans, img
    return img2


def readTiff(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    projection = dataset.GetProjection()  # 投影
    geotrans = dataset.GetGeoTransform()  # 几何信息
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    # print(im_bands)
    img_array = dataset.ReadAsArray().astype(np.float32)
    img_array = img_array[:3, :, :]
    imgScale = img_array / 10000
    img = imgScale * 255
    img = np.transpose(img, (1, 2, 0))
    return im_width, im_height, im_bands, projection, geotrans, img


def readTiff_uint16(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    projection = dataset.GetProjection()  # 投影
    geotrans = dataset.GetGeoTransform()  # 几何信息
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    # print(im_bands)
    img_array = dataset.ReadAsArray()

    img = np.transpose(img_array, (1, 2, 0))
    return img


def writeTif(tiff_file, im_proj, im_geotrans, data_array):
    if 'int8' in data_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_array.dtype.name:
        datatype = gdal.GDT_Int16
    else:
        datatype = gdal.GDT_Float32

    if len(data_array.shape) == 3:
        im_bands, im_height, im_width = data_array.shape
    else:
        im_bands, (im_height, im_width) = 1, data_array.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(tiff_file, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data_array)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data_array[i])

    del dataset
