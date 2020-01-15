import os


"""
Generate list of filenames (for folder) without extension
"""
def get_all_filenames(folder):
    return list(map(lambda s: s.split('.')[0], os.listdir(folder)))