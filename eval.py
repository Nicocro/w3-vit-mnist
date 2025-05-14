import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dataset import MNIST_2x2
from model.model import ImageToDigitTransformer
from utils.tokenizer import START, FINISH, decode

# device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# config
VOCAB_SIZE = 13
MAX_LEN = 5  # length of decoder input: [<start>, d1, d2, d3, d4]
SEQ_LEN = 4  # number of predicted digits

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_dataset = MNIST_2x2(mnist_test, seed=42)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ImageToDigitTransformer(vocab_size=VOCAB_SIZE).to(device)
model.load_state_dict(torch.load("checkpoints/transformer_mnist.pt", map_location=device))
model.eval()

# Evaluation Loop
correct_sequences = 0
digit_correct = 0
digit_total = 0

with torch.no_grad():
    loop = tqdm(test_loader, desc="Evaluating", leave=False)

    for image, _, target_ids in loop:
        image = image.to(device)
        target_ids = target_ids.squeeze(0).tolist()[:-1]  # remove <finish>

        decoded = [START]
        for _ in range(SEQ_LEN):
            input_ids = torch.tensor(decoded, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(image, input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            decoded.append(next_token)
            if next_token == FINISH:
                break

        pred = decoded[1:][:SEQ_LEN]
        target = target_ids

        if pred == target:
            correct_sequences += 1
        digit_correct += sum(p == t for p, t in zip(pred, target))
        digit_total += len(target)

        seq_acc = 100.0 * correct_sequences / (digit_total // SEQ_LEN)
        digit_acc = 100.0 * digit_correct / digit_total
        loop.set_postfix(seq_acc=f"{seq_acc:.2f}%", digit_acc=f"{digit_acc:.2f}%")


# final results
total_samples = len(test_loader)
seq_acc = 100.0 * correct_sequences / total_samples
digit_acc = 100.0 * digit_correct / digit_total

print(f"\nFinal Evaluation Results:")
print(f"  Sequence accuracy: {seq_acc:.2f}%")
print(f"  Per-digit accuracy: {digit_acc:.2f}%")