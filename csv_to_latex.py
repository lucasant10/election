import csv

lines = {}
with open('/Volumes/Data/eleicoes/validation_final.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
    for row in spamreader:
        model_name = row[0].replace ('_', ' ')
        
        source = row[1].strip().upper()
        
        accuracy = float(row[2])

        p_non_pol = float(row[3])
        p_pol = float(row[4])

        r_non_pol = float(row[5])
        r_pol = float(row[6])
        
        f1_non_pol = float(row[7])
        f1_pol = float(row[8])
        
        
        s_non_pol = float(row[9])
        s_pol = float(row[10])
        
        f1_macro = float(row[11])
        recall_macro = float(row[12])
        precision_macro = float(row[13])

        mean_auc = float (row[14])
        std_auc = float(row[15])
        
        ff1 = float (row[16])
        recall = float (row[17])
        precision = float (row[18])

        if source not in lines:
            lines [source] = list()
        
        lines [source].append([source, model_name, accuracy * 100, p_pol * 100, r_pol * 100, f1_pol * 100, p_non_pol * 100, r_non_pol * 100, f1_non_pol * 100, f1_macro * 100])


        

for source in lines:
    first = True
    for line in lines[source]:
        
        if first:
            first = False
            print ('%s & \multicolumn{1}{l|}{%s} & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ \\\\' % tuple(line))
        else:
            print (' & \multicolumn{1}{l|}{%s} & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ & $%0.2f$ \\\\' % tuple(line[1:]))