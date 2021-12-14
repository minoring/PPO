import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def to_tensor(array):
    return torch.from_numpy(array).float().to(device)


def to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()
