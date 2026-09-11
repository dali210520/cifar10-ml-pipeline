import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, data_path="./CIFAR-10"):
    """
    Lädt den CIFAR-10 Datensatz, normalisiert die Bilder
    und erstellt Train-, Validierungs- und Test-Loader.
    """

    # Transformation: Tensor + Normalisierung
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # Trainingsdaten (50.000 Bilder)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    # Testdaten (10.000 Bilder)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    # 80/20 Split für Training und Validierung
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )

    # DataLoader erzeugen
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
