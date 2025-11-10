import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torchvision.models import resnet18, ResNet18_Weights
import json, os

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder("data/train", transform=transform)
test_ds  = datasets.ImageFolder("data/test", transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64)


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(15):
    model.train()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward(); opt.step()
    print(f"Epoch {epoch+1} done")

model.eval()
preds, truths = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        preds += out.argmax(1).cpu().tolist()
        truths += lbls.cpu().tolist()

le = LabelEncoder()
le.fit(range(len(train_ds.classes)))
print(classification_report(truths, preds, labels=range(len(train_ds.classes)), target_names=train_ds.classes))

torch.save(model.state_dict(), "outputs/emotion_model.pth")
with open("outputs/predictions.json", "w") as f:
    json.dump({"labels": train_ds.classes,
               "preds": preds[:20],
               "truths": truths[:20]}, f)
