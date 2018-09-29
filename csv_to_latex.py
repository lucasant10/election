import csv
with open('/Volumes/Data/eleicoes/report/validation_balanced.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
    for row in spamreader:
        model_name = row[0].replace ('_', ' ').capitalize()
        source = row[1].replace ('.politics', '').replace ('_', ' ').upper()
        precision_mean = float(row[2])
        precision_std = float(row[3])
        recall_mean = float(row[4])
        recall_std = float(row[5])
        f1_mean = float(row[6])
        f1_std = float(row[7])

        print ('%s & %s & $%0.3f$ & $\pm%0.3f$ & $%0.3f$ & $\pm%0.3f$ & $%0.3f$ & $\pm%0.3f$ \\\\ \hline' % (model_name, source, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std))