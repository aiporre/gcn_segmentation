import csv


def csv_to_dict(filename, delimeter, has_header=False, key_col=0, item_col=1):
    """
    CSV file has two columns
    """
    values = {}
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimeter)
        header_skipped = False
        for row in csvreader:
            if has_header and not header_skipped:
                header_skipped = True
                continue  # _ = csvreader.next()
            values[row[key_col]] = row[item_col]
    return values


def dict_to_csv(filename, values, delimiter=',', index=None):
    header = [""] + list(values.keys())
    columns = list(values.values())

    def all_equal(vv):
        return all(x == vv[0] for x in vv)

    lengths = [len(c) for c in columns]
    length = max(lengths)
    assert all_equal(lengths), 'Dictionary input in fcn dict_to_csv must have the same number of entries per key'

    if index is None:
        index = list(range(length))
    dict_values = [dict(zip(header, vv)) for vv in zip(index, *columns)]
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=delimiter)
            writer.writeheader()
            for data in dict_values:
                writer.writerow(data)
    except IOError:
        print("I/O error")
