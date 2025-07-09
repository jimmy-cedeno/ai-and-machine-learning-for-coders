import os
import urllib.request
import zipfile

def download(url_file, name, directory):
    urllib.request.urlretrieve(url_file, name)

    zip_ref = zipfile.ZipFile(name, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()



url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
file_name = "../horse-or-human.zip"
training_dir = 'CNNHorseOrHuman/horse-or-human/training/'

download(url, file_name, training_dir)

url_validation = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
file_name_validation = "../validation-horse-or-human.zip"
training_dir_validation = 'CNNHorseOrHuman/horse-or-human/validation/'

download(url_validation, file_name_validation, training_dir_validation)

# urllib.request.urlretrieve(url, file_name)
#
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()