import util_functions
from random import shuffle
import os
import itertools
from multiprocessing import Pool
import psutil



def save_spectrograms(wave_path, save_root_dir, label_map):
	wave_id = util_functions.get_file_name_without_extension(wave_path)
	save_path = os.path.join(save_root_dir, "{}.png".format(wave_id))
	if label_map != None:
		wave_label = label_map[wave_id]
		save_dir = os.path.join(save_root_dir, wave_label)
		save_path = os.path.join(save_dir, "{}.png".format(wave_id))
	spectrogram_img = util_functions.wave_path_to_rgb_spectrogram_img(wave_path)
	spectrogram_img.save(save_path)



def main():
	train_data_dir = "g2net-gravitational-wave-detection\\train"
	test_data_dir = "g2net-gravitational-wave-detection\\test"
	train_labels_path = "g2net-gravitational-wave-detection\\training_labels.csv"
	test_proportion = 0.2
	train_test_save_dir = "spectrograms\\train\\test"
	train_train_save_dir = "spectrograms\\train\\train"
	test_save_dir = "spectrograms\\test"

	train_wave_paths = util_functions.get_all_subfile_paths_in_directory(train_data_dir)
	shuffle(train_wave_paths)
	train_test_count = int(round(len(train_wave_paths)*test_proportion))
	train_test_wave_paths = train_wave_paths[:train_test_count]
	train_train_wave_paths = train_wave_paths[train_test_count:]
	test_wave_paths = util_functions.get_all_subfile_paths_in_directory(test_data_dir)

	label_map = util_functions.csv_to_dict(train_labels_path)

	num_physical_cores = psutil.cpu_count(logical=False)
	cores_to_use = num_physical_cores - 2

	# # Save train test spectrograms
	# with Pool(cores_to_use) as pool:
	# 	arguments = zip(train_test_wave_paths, itertools.repeat(train_test_save_dir), itertools.repeat(label_map))
	# 	pool.starmap(save_spectrograms, arguments)
	
	# # Save train train spectrograms
	# with Pool(cores_to_use) as pool:
	# 	arguments = zip(train_train_wave_paths, itertools.repeat(train_train_save_dir), itertools.repeat(label_map))
	# 	pool.starmap(save_spectrograms, arguments)
	
	# Save train test spectrograms
	with Pool(cores_to_use) as pool:
		arguments = zip(test_wave_paths, itertools.repeat(test_save_dir), itertools.repeat(None))
		pool.starmap(save_spectrograms, arguments)

	# util_functions.wave_path_to_rgb_spectrogram_img("g2net-gravitational-wave-detection\\test\\0\\0\\2\\0021f9dd71.npy").show()



if __name__ == "__main__":
	main()