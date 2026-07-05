import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# 1. Definisikan Arsitektur Model (Wajib SAMA PERSIS dengan Kaggle)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x): return self.linear(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )
    def forward(self, x): return self.linear(x).view(-1, 1, 28, 28)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, x): return self.decoder(self.encoder(x))

if __name__ == "__main__":
    # 2. Pengaturan Argumen Terminal
    parser = argparse.ArgumentParser(description="Rekonstruksi Citra Fashion-MNIST via Terminal")
    parser.add_argument("--model", type=str, required=True, help="Nama file model .pth")
    parser.add_argument("--latent_dim", type=int, required=True, help="Ukuran dimensi latent model")
    parser.add_argument("--index", type=int, default=0, help="Indeks gambar yang ingin direkonstruksi")
    args = parser.parse_args()

    # Checking file model
    if not os.path.exists(args.model):
        print(f"Error: File model '{args.model}' tidak ditemukan di folder ini!")
        exit()

    # 3. Load Model
    model = Autoencoder(args.latent_dim)
    
    # Cek apakah targetnya folder atau file biasa
    if os.path.isdir(args.model):
        # Jika berupa folder (seperti autoencoder_dim32), arahkan langsung ke data.pkl di dalamnya
        state_dict_path = os.path.join(args.model, "data.pkl")
        print(f"-> Mendeteksi folder model, memuat weights dari: {state_dict_path}")
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    else:
        # Jika berupa file .pth standar
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        
    model.eval()
    print(f"-> Sukses memuat model {args.model} dengan Latent Dim: {args.latent_dim}")
    # 4. Load Dataset Fashion-MNIST (untuk testing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("-> Memuat dataset Fashion-MNIST...")
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    
    # Ambil gambar berdasarkan indeks
    img, label = test_dataset[args.index]
    
    # 5. Jalankan Rekonstruksi (Inferensi)
    with torch.no_grad():
        # Tambahkan dimensi batch awal: [1, 1, 28, 28]
        recon_img = model(img.unsqueeze(0)).squeeze(0)

    # Denormalisasi gambar kembali ke rentang [0, 1] agar bisa disimpan dengan benar
    img = (img + 1) / 2
    recon_img = (recon_img + 1) / 2

    # 6. Simpan Hasil Gambar Berdasarkan Instruksi Tugas
    plt.imsave("original.png", img.squeeze().numpy(), cmap='gray')
    plt.imsave("reconstructed.png", recon_img.squeeze().numpy(), cmap='gray')

    # Buat perbandingan berdampingan (comparison.png)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img.squeeze().numpy(), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(recon_img.squeeze().numpy(), cmap='gray')
    axes[1].set_title(f"Reconstructed (Dim {args.latent_dim})")
    axes[1].axis('off')
    
    plt.savefig("comparison.png", bbox_inches='tight')
    print("-> Sukses! File 'original.png', 'reconstructed.png', dan 'comparison.png' telah disimpan di folder.")