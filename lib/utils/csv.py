import csv

def csv_to_dict(filename, delimeter):
    '''
    CSV file has two columns
    '''
    values = {}
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimeter)
        for row in csvreader:
            values[row[0]] = row[1]
    return values