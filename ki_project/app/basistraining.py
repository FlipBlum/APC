from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from simpleCNN import SimpleCNN  # Importieren der Modelldefinition

# Erstellen Sie ein neues Experiment-Objekt
experiment = Experiment(api_key="0NPgf4vYBtZjxKoE50bCNAbuL", project_name="MachineVision")
epochenanzahl = int(input("Geben Sie die Anzahl der Epochen ein: "))

def main():
    # Daten-Vorbereitung
    transform = transforms.Compose([transforms.Resize((180, 180)), transforms.ToTensor()])
    train_data = datasets.ImageFolder("/Users/philippblum/Desktop/coding/ki_project/static/images/train", transform=transform)
    print(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_data = datasets.ImageFolder("/Users/philippblum/Desktop/coding/ki_project/static/images/validate", transform=transform)
    print(val_data)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    model = SimpleCNN()

    # Verlust und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(epochenanzahl):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Log the training loss with comet_ml
            experiment.log_metric("train_loss", loss.item())
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        # Log validation loss and accuracy
        experiment.log_metric("val_loss", val_loss / len(val_loader))
        print("Accuracy: ", (correct / total)*100, "%")
        experiment.log_metric("val_accuracy", correct / total)
      
        print(f"Ein Epochendurchlauf ist fertig!")
      
    torch.save(model.state_dict(), "ringprediciton.pth")

    # End the experiment
    experiment.end()

if __name__ == "__main__":
    main()
