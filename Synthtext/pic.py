
import os

f = open('labels.txt','w')
directory = 'pictures'
for filename in os.listdir(directory):
    f.writelines(os.path.join(directory, filename))
    f.writelines('\n')
