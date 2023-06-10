from model import NCA

# load medmnist data
from medmnist.info import INFO



def load_mnist_data():
    # load medmnist data
    info = INFO['octmnist']
    data = info['dataloader'](
        root=info['root'],
        split='train',
        transform=info['transform'],
        download=True
    )

    return data


def train():
    pass