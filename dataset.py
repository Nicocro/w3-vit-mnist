import torch 
from torch.utils.data import Dataset 
from torchvision import datasets, transforms

from utils.tokenizer import prepare_decoder_labels, encode, decode

class MNIST_2x2(Dataset):
    def __init__(self, base_dataset, transform=None, seed=42):
        self.base_dataset = base_dataset         
        self.transform = transform 
        self.length = len(base_dataset)

        torch.manual_seed(seed)
        self.index_map = [
            torch.randint(0, self.length, (4,))
            for _ in range(self.length)
        ]

    def __len__(self):
        return self.length 
    
    def __getitem__(self, idx):
        indices = self.index_map[idx]

        images = [self.base_dataset[i][0] for i in indices]
        top_row = torch.cat([images[0], images[1]], dim=2)
        bottom_row = torch.cat([images[2], images[3]], dim=2)
        grid_image = torch.cat([top_row, bottom_row], dim=1)

        labels = [self.base_dataset[i][1] for i in indices]
        decoder_input_ids, decoder_target_ids = prepare_decoder_labels(labels)
        decoder_input = torch.tensor(decoder_input_ids, dtype=torch.long)
        decoder_target = torch.tensor(decoder_target_ids, dtype=torch.long)

        return grid_image, decoder_input, decoder_target
            
# test the dataset and visualize a few samples 
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_dataset = MNIST_2x2(mnist_train, seed=42)
    test_dataset = MNIST_2x2(mnist_test, seed=42)


    def show_grid_image(grid_tensor, decoder_target):
        # Undo normalization for visualization
        img = grid_tensor.clone()
        img = img * 0.3081 + 0.1307
        img = img.squeeze().numpy()

        # Decode token IDs into digit strings
        digits = decode(decoder_target.tolist()[:-1])  # Remove <finish> for display
        label_str = " ".join(digits)

        plt.imshow(img, cmap="gray")
        plt.title(f"Digits: {label_str}")
        plt.axis("off")
        plt.show()

    # Visualize a few samples
    for i in range(3):
        grid_image, decoder_input, decoder_target = train_dataset[i]
        show_grid_image(grid_image, decoder_target)