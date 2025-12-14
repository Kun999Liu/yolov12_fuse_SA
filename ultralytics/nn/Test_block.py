# -*- coding: utf-8 -*-
# @Time    : 2025/12/14 20:55
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : Test_block.py
# @Software: PyCharm

"""
Describe:
"""

from modules.block import SpectralStem


if __name__ == '__main__':
    stem = SpectralStem(c1=4, c2=64, k=3, s=2)
    print(stem)