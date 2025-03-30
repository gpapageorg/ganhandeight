import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for Tanh activation
])

# Load MNIST dataset (without transform initially)
train_set = datasets.MNIST(root='./data', train=True, download=True)

# Filter only the '8's from MNIST
indices = [i for i, label in enumerate(train_set.targets) if label == 8]

# Custom dataset wrapper to apply transformations
class MNISTSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

# Create dataset with only '8's and apply transforms
train_set_8 = MNISTSubset(train_set, indices, transform=transform)
train_loader = DataLoader(train_set_8, batch_size=64, shuffle=True)

# ========================
#        Generator
# ========================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # Scale output to [-1, 1] for consistency with normalized images
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), 1, 28, 28)  # Reshape to (batch, channels, height, width)

# ========================
#      Discriminator
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability (real or fake)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        return self.model(x)

# Instantiate models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# # Loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 150
real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)

        # ========================
        #    Train Discriminator
        # ========================
        discriminator.zero_grad()
        
        # Real images
        batch_size = images.size(0)
        label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output_real = discriminator(images).view(-1)
        loss_d_real = criterion(output_real, label_real)
        loss_d_real.backward()

        # Fake images
        noise = torch.randn(batch_size, 100, device=device)
        fake_images = generator(noise)
        label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
        output_fake = discriminator(fake_images.detach()).view(-1)
        loss_d_fake = criterion(output_fake, label_fake)
        loss_d_fake.backward()

        optimizer_d.step()

        # ========================
        #      Train Generator
        # ========================
        generator.zero_grad()
        label_gen = torch.full((batch_size,), real_label, dtype=torch.float, device=device)  # Generator wants to fool D
        output_gen = discriminator(fake_images).view(-1)
        loss_g = criterion(output_gen, label_gen)
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {loss_d_real.item() + loss_d_fake.item():.4f} | Loss G: {loss_g.item():.4f}")

    # Display generated images every 10 epochs
    # if (epoch + 1) % 10 == 0:
    #     with torch.no_grad():
    #         noise = torch.randn(64, 100, device=device)
    #         fake_images = generator(noise)
    #         grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
    #         plt.figure(figsize=(8, 8))
    #         plt.imshow(grid.permute(1, 2, 0).cpu())
    #         plt.axis('off')
    #         plt.show()

# Generate and show final images after training

generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))

# Set to evaluation mode (important for inference)
generator.eval()
discriminator.eval()

with torch.no_grad():
    noise = torch.randn(64, 100, device=device)
    fake_images = generator(noise)
    fake_images = -fake_images  # Inverts pixel values
    fake_images = (fake_images + 1) / 2  # Re-normalize from [-1,1] to [0,1]
    grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
    plt.axis('off')
    plt.show()
torch.save(generator.state_dict(), "generator"+str(num_epochs)+".pth")
torch.save(discriminator.state_dict(), "discriminator"+str(num_epochs)+".pth")
print("Models saved successfully!")