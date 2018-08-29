import os
import re

path = 'models'

def sort_models(path):
    list = []
    for x in os.listdir(path):
        try:
            file = open(path + '\\' + x + '\\log.txt')
            acc = re.findall("\d+\.\d+", file.read())
            # print(acc[0])
            list.append((float(acc[0]), x))
        except Exception:
            pass
    list.sort()
    print(list[:20])

sort_models(path)