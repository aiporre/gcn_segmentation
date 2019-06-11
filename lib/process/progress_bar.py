# Print iterations progress
import sys


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('%s [%s] %s%s ...%s \r ' % (prefix, bar, percent, '%', suffix))
    sys.stdout.flush()
    if iteration == total:
        print()


if __name__ == '__main__':
    items = list(range(0, 57))
    l = len(items)
    from time import sleep

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i, item in enumerate(items):
        # Do stuff...
        sleep(0.1)
        # Update Progress Bar
        printProgressBar(i+1, l, prefix='Progress:', suffix='Complete', length=50)

