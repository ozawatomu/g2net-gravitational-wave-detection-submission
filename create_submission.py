import torch
from torchvision import transforms
from sys import stdout
import util_functions
from PIL import Image
import numpy as np
import csv
import my_models



def image_loader(loader, image_name, device=torch.device('cuda')):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device=device)



def main():
	# Get all test spectrogram paths
	test_imgs_path = "spectrograms\\test"
	test_img_paths = util_functions.get_all_subfile_paths_in_directory(test_imgs_path)

	# Prepare model
	comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.load("trained-models\\1626928232\\28.pt").to(comp_device)
	model.eval()

	# Make predictions
	predictions = {}
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226])
	])
	with torch.no_grad():
		for i, test_img_path in enumerate(test_img_paths):
			if (i + 1) % 1000 == 0:
				stdout.write("{}/{}\n".format(i + 1, len(test_img_paths)))
			x = image_loader(transform, test_img_path, comp_device)
			scores = model(x).cpu().detach().numpy()
			prediction = np.argmax(scores)
			id = util_functions.get_file_name_without_extension(test_img_path)
			predictions[id] = prediction
	
	# Save Predictions
	with open("submission.csv", "w", newline="") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["id", "target"])

		for key, value in predictions.items():
			csv_writer.writerow([key, value])



if __name__ == "__main__":
	main()