import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class DM_Dataset(Dataset):
    def __init__(self, X_input, Y_input):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.Y_input[idx]