import os

def get_npy_files(path='./', extension='npy'):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(file))
    return files