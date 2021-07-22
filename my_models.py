import torch.nn as nn
import torch.nn.functional as F
import torch



class Model1(nn.Module):

	def __init__(self):
		super().__init__()
		self.name = "Model 1"
		self.conv1 = nn.Conv2d(3, 8, 3, 1, padding='same', padding_mode='zeros')
		self.conv2 = nn.Conv2d(8, 16, 3, 1, padding='same', padding_mode='zeros')
		self.pool1 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(17408, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 2)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)

		x = x.view(-1, 17408)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.log_softmax(self.fc3(x), dim=1)

		return x



class Model2(nn.Module):

	def __init__(self):
		super().__init__()
		self.name = "Model 2"
		self.conv1 = nn.Conv2d(3, 64, 3, 1, padding='same', padding_mode='zeros')
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(64, 128, 3, 1, padding='same', padding_mode='zeros')
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(128, 256, 3, 1, padding='same', padding_mode='zeros')
		self.pool3 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(256*8*8, 1024)
		self.fc2 = nn.Linear(1024, 256)
		self.fc3 = nn.Linear(256, 2)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = F.relu(self.conv3(x))
		x = self.pool3(x)

		x = x.view(-1, 256*8*8)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.log_softmax(self.fc3(x), dim=1)

		return x



class