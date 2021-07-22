from os import walk, makedirs, devnull
from os.path import basename, splitext, join, exists
import sys
import numpy as np
import torch
from nnAudio.Spectrogram import CQT1992v2
from PIL import Image
import csv
import logging
logging.disable = True



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def get_all_subfile_paths_in_directory(root_dir):
	file_paths = []
	for path, sub_dirs, files in walk(root_dir):
		for file_name in files:
			file_path = join(path, file_name)
			file_paths.append(file_path)
	return file_paths



def get_file_name_without_extension(datum_file_path):
	datum_file_name = basename(datum_file_path)
	datum_file_raw_name = splitext(datum_file_name)[0]
	return datum_file_raw_name



def normalize_array(array, value_from=0, value_to=1):
	max_value = array.max()
	min_value = array.min()

	normalized_array = (value_to - value_from)*(array - min_value)/(max_value - min_value) + value_from
	return normalized_array



def wave_path_to_rgb_spectrogram_img(wave_path, sr=2048, fmin=20, fmax=1024, hop_length=64):
	wave_arrays = np.load(wave_path)

	spectrogram_arrays = []
	for wave_array in wave_arrays:
		spectrogram_array = wave_to_spectrogram_array(wave_array, sr, fmin, fmax, hop_length)
		spectrogram_array_normalized = normalize_array(spectrogram_array, 0, 255)
		spectrogram_arrays.append(spectrogram_array_normalized)

	spectrogram_rgb_img_array = np.dstack(spectrogram_arrays).astype(np.uint8)
	return Image.fromarray(spectrogram_rgb_img_array)



def wave_to_spectrogram_array(wave_array, sr=2048, fmin=20, fmax=1024, hop_length=64):
	torch.from_numpy(wave_array)
	wave_tensor = torch.from_numpy(wave_array).float()
	transform = None
	with HiddenPrints():
		transform = CQT1992v2(sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
	spectrogram_tensor = transform(wave_tensor)
	spectrogram_array = np.array(spectrogram_tensor)
	return spectrogram_array.reshape(spectrogram_array.shape[1], -1)



def csv_to_dict(csv_path, contains_header=True):
	csv_dict = {}
	with open(csv_path) as csv_file:
		csv_reader = csv.reader(csv_file)
		if contains_header:
			next(csv_reader)
		for row in csv_reader:
			csv_dict[row[0]] = row[1]
	return csv_dict



def create_dir_if_not_exist(directory):
	if not exists(directory):
		makedirs(directory)