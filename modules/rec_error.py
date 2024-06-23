# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:15:24 2024

@author: s183753
"""

def rec_error(rec_1, rec_2):
    import numpy as np
    rec_error = np.abs(rec_1 - rec_2)
    return rec_error