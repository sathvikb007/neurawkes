import joblib
import os
import sys

def convert(dir):
    """
    Convert all files in dir from .pickle to .pkl
    """                                                   
    for file in os.listdir(dir):
        if file.endswith(".pickle"):
            file_name = os.path.join(dir, file)
            print(file_name)
            joblib.dump(joblib.load(file_name), file_name.replace(".pickle", ".pkl"), protocol= 2)
            os.remove(file_name)

def solve(x):
    """
        find websites from URL x
    """

convert('./data/data_epic/Task6/')

    