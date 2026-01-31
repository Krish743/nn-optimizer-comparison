from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=64):
    def flatten(x):
        return x.view(-1)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(flatten)
        ])

    train_data = datasets.MNIST(
            root= './data',
            train = True,
            download= True,
            transform= transform
        )

    test_data = datasets.MNIST(
        root = './data',
        train=False,
        download = True,
        transform = transform)

    train_loader = DataLoader(batch_size=batch_size, dataset= train_data, shuffle= True, pin_memory= True, num_workers= 0)
    test_loader = DataLoader(batch_size=batch_size, dataset= test_data, shuffle= False, pin_memory= True, num_workers= 0)

    return train_loader, test_loader