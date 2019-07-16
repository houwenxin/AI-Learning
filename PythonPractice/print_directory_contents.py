# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:04:36 2019

@author: houwenxin
"""

def print_directory_contents(sPath):
    import os
    for sChild in os.listdir(sPath):
        sChildPath = os.path.join(sPath, sChild)
        if os.path.isdir(sChildPath):
            print_directory_contents(sChildPath)
        else:
            print(sChildPath)
            
def test(path):
    print_directory_contents(path)

if __name__ == "__main__":
    path = "../"
    test(path)