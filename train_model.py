import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from sys import stdout
from os.path import join
import pickle
from matplotlib import pyplot as plt
import util_functions
from my_models import ConvModel
import warnings



def main():
	# https://www.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/learn/lecture/14838210#overview

	# Hide warnings
	show_warnings = False
	if not show_warnings:
		warnings.filterwarnings("ignore")

	# Prepare datasets
	train_data_folder = "spectrograms\\train\\train"
	test_data_folder = "spectrograms\\train\\test"

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226])
	])
	train_dataset = datasets.ImageFolder(train_data_folder, transform=transform)
	test_dataset = datasets.ImageFolder(test_data_folder, transform=transform)

	# Prepare model
	comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = ConvModel().to(comp_device)
	model.train()

	# Set hyperparameters
	learning_rate = 3e-4
	batch_size = 32
	num_epochs = 32

	# Create loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# Create data loaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)

	# Train model
	start_time = time.time()

	results_save_dir = "trained-models\\{}".format(int(start_time))
	util_functions.create_dir_if_not_exist(results_save_dir)

	epochs = []
	train_losses = []
	test_losses = []
	test_scores = []

	for epoch_i in range(num_epochs):
		epochs.append(epoch_i + 1)
		stdout.write("Epoch: {}/{}\t".format(epoch_i + 1, num_epochs))

		# model.train()
		for batch_i, (train_x, train_y) in enumerate(train_loader):
			train_x = train_x.to(comp_device)
			train_y = train_y.to(comp_device)

			pred_y = model(train_x)
			loss = criterion(pred_y, train_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		train_losses.append(loss.cpu().detach().numpy())
		stdout.write("Train Loss: {:.2f}\t".format(train_losses[-1]))
		torch.save(model, join(results_save_dir, "{}.pt".format(epoch_i + 1)))

		# model.eval()
		num_correct = 0
		num_samples = 0
		with torch.no_grad():
			for batch_i, (test_x, test_y) in enumerate(test_loader):
				test_x = test_x.to(comp_device)
				test_y = test_y.to(comp_device)

				pred_y = model(test_x)
				_, predictions = pred_y.max(1)
				num_correct += (predictions == test_y).sum()
				num_samples += predictions.size(0)
		
		loss = criterion(pred_y, test_y)
		test_losses.append(loss.cpu().detach().numpy())
		score = num_correct/num_samples
		test_scores.append(score)
		stdout.write("Test Loss: {:.2f}\t".format(test_losses[-1]))
		stdout.write("Test Score: {:.4f}\n".format(test_scores[-1]))

	total_time = time.time() - start_time
	stdout.write("Total time: {} minutes\n".format(total_time/60))

	# Save training details
	training_details_path = join(results_save_dir, "training-details.txt")
	with open(training_details_path, "w") as txt_file:
		txt_file.write("Total training time: {} minutes.\n".format(total_time/60))
		for i in range(len(epochs)):
			txt_file.write("Epoch: {},\tTrain loss: {:.3f},\tTest loss: {:.3f},\tTest Score: {:.4f}\n".format(epochs[i], train_losses[i], test_losses[i], test_scores[i]))

	# Save pickles
	with open(join(results_save_dir, "epochs.pickle"), "wb") as pickle_file:
		pickle.dump(epochs, pickle_file)
	with open(join(results_save_dir, "train_losses.pickle"), "wb") as pickle_file:
		pickle.dump(train_losses, pickle_file)
	with open(join(results_save_dir, "test_losses.pickle"), "wb") as pickle_file:
		pickle.dump(test_losses, pickle_file)
	with open(join(results_save_dir, "test_scores.pickle"), "wb") as pickle_file:
		pickle.dump(test_scores, pickle_file)
	
	# Plot losses
	plt.plot(epochs, train_losses)
	plt.plot(epochs, test_losses, 'r')
	plt.show()

	plt.plot(epochs, test_scores)
	plt.show()



if __name__ == "__main__":
	main()