import glob
import re

for file in glob.iglob('./validation_set/*ohc.txt'):
    ohc_file = open(file, 'r')
    ohc2 = open(file[:-4] + '_ohc2.txt', 'w+')
    for line in ohc_file.readlines():
        x_loc = re.search('\((.+?)\,', line)
        if x_loc:
            x = x_loc.group(1)
        y_loc = re.search('\s(.+?)\)', line)
        if y_loc:
            y = y_loc.group(1)
        location = (int(x)*2.724, int(y)*2.724)
        ohe_data = [location, 1, 0]
        ohc2.write(str(ohe_data) + '\n')
    ohc_file.close()
    ohc2.close()