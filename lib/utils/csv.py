import csv

def csv_to_dict(filename, delimeter, has_header=False, key_col=0, item_col=1):
    '''
    CSV file has two columns
    '''
    values = {}
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimeter)
        header_skipped = False
        for row in csvreader:
            if has_header and not header_skipped:
                header_skipped = True
                continue #_ = csvreader.next()
            values[row[key_col]] = row[item_col]
    return values