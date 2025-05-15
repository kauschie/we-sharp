import unicodedata
import os

def normalize_filenames(path):
    for filename in os.listdir(path):
        new_name = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
        os.rename(os.path.join(path, filename), os.path.join(path, new_name))


