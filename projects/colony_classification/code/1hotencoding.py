import glob

for grande in glob.iglob('./*grande.txt'):
    grande_file = open(grande, 'r')
    petite_file = open('./' + grande[2:-11] + '_petite.txt', 'r')
    ohc_file = open(grande[2:-11] + '_ohc2.txt', 'w+')
    for line in grande_file.readlines():
        line = line.rstrip('\n')
        location = tuple([int(i)*2.724 for i in line.split(',')])
        ohe_data = [location, 1, 0]
        ohc_file.write(str(ohe_data) + '\n')
    for line in petite_file.readlines():
        line = line.rstrip('\n')
        location = tuple([int(i)*2.724 for i in line.split(',')])
        ohe_data = [location, 0, 1]
        ohc_file.write(str(ohe_data) + '\n')
    grande_file.close()
    petite_file.close()
    ohc_file.close()
        