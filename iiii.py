import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
from tqdm import tqdm
from torchvision import transforms

# Assumi che SatCLIP sia definito in un altro file cshiamato 'model.py'
from model import SatCLIP

class SatCLIPClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_path):
        super(SatCLIPClassifier, self).__init__()
        
        # Carica il modello SatCLIP pre-addestrato
        self.satclip = SatCLIP.load_from_checkpoint(pretrained_model_path)
        
        # Blocca i parametri del modello SatCLIP
        for param in self.satclip.parameters():
            param.requires_grad = False
        
        # Sblocca i parametri dell'encoder visivo
        for param in self.satclip.visual.parameters():
            param.requires_grad = True
        
        # Aggiungi un classificatore lineare
        self.classifier = nn.Linear(self.satclip.embed_dim, num_classes)

    def forward(self, images):
        # Codifica le immagini
        image_features = self.satclip.encode_image(images)
        
        # Normalizza le feature (come fatto in SatCLIP)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Classifica
        outputs = self.classifier(image_features)
        return outputs

class EuroSATDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.data = []
        self.targets = []
        
        for cls in self.classes:
            class_path = os.path.join(root, cls)
            for img_name in os.listdir(class_path):
                self.data.append(os.path.join(class_path, img_name))
                self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
        
        # Valutazione
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy on test set: {100 * correct / total}%')

def main():
    # Impostazioni
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_path = 'path/to/pretrained/satclip/model.ckpt'
    eurosat_root = '/work/tesi_acastagni/EuroSAT'
    num_epochs = 10
    batch_size = 64

    # Preparazione del dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EuroSATDataset(root=eurosat_root, transform=transform)
    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Creazione e preparazione del modello
    model = SatCLIPClassifier(num_classes, pretrained_model_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Addestramento e valutazione
    train_and_evaluate(model, trainloader, testloader, criterion, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()