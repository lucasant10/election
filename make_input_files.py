import csv

politics = list()
non_politics = list ()
for i in range (1, 5):
    with open('/Volumes/Data/eleicoes/input/fold'+str(i)+'.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
        for row in spamreader:
            text = row[0]
            is_political = row[1]

            if int(is_political):
                politics.append (row)
            else:
                non_politics.append (row)





with open('/Volumes/Data/eleicoes/input/S2_politics.txt', 'w') as txt_politics:
    for item in politics:
        txt_politics.write (item[0] + '\r\n')


with open('/Volumes/Data/eleicoes/input/S2_non-politics.txt', 'w') as txt_non_politics:
    for item in non_politics:
        txt_non_politics.write (item[0] + '\r\n')

print ('Politics entries: %s' % (len(politics)))
print ('Non Politics entries: %s' % (len(non_politics)))

