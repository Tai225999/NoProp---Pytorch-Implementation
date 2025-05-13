from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(batch_size=128, data_dir='./data'):
    """
    Loads and preprocesses the CIFAR-10 dataset.
    
    This function:
    1. Downloads CIFAR-10 if not present
    2. Applies data augmentation to training set
    3. Splits training data into train/validation sets
    4. Creates DataLoaders for train/val/test sets
    
    Args:
        batch_size (int, optional): Batch size for all DataLoaders. Defaults to 128.
        data_dir (str, optional): Directory to store/load the data from. Defaults to './data'.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing:
            - train_loader: DataLoader for training set
            - val_loader: DataLoader for validation set
            - test_loader: DataLoader for test set
    
    Note:
        - Training set is split into 39,936 training and 64 validation samples
        - Data augmentation (random crop, horizontal flip) is applied to training set
        - Images are normalized using CIFAR-10 mean and std values
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # Download and load the training data
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    # Split the dataset into training and validation sets
    train_size = 39936 # Largest multiple of 128 under 40000
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2 # Adjust num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2 # Adjust num_workers
    )

    # Download and load the test data
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2 # Adjust num_workers
    )

    print("CIFAR-10 dataset loaded.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example usage:
    train_loader, test_loader = load_cifar10(batch_size=64)
    # Get some samples
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("Image batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)