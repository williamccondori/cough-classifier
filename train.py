import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

""" Definicion """
""" ********** """

# Se define la transformacion.
data_transforms = transforms.Compose(
    [transforms.Resize((160, 160)), transforms.ToTensor()])

# Se define el dispositivo.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" Carga de datos """
""" ************** """

train_path = ''
eval_path = ''

# Se leen los datos.
train_data = torchvision.datasets.ImageFolder(
    train_path, transform=data_transforms)
eval_data = torchvision.datasets.ImageFolder(
    eval_path, transform=data_transforms)

# Se defninen los data loaders.
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=4)
eval_loader = torch.utils.data.DataLoader(
    eval_data, batch_size=32, shuffle=True, num_workers=4)

data_loaders = {'train': train_loader, 'eval': eval_loader}


def train(model, data_loaders, criterion, optimizer, scheduler, epochs=100):
    sizes = {'train': len(train_data), 'eval': len(eval_data)}
    for epoch in range(epochs):

        # Epoch.
        for phase in ['train', 'eval']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in data_loaders[phase]:

                # Carga los datos al dispositivo actual.
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # Fordward.
                    outputs = model(inputs)
                    _, prediction = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':

                        # Backward.
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(prediction == labels.data)

            # Epoch phase result.
            epoch_loss = running_loss / sizes[phase]
            epoch_accuracy = running_corrects / sizes[phase]
            print('epoch:{} phase:{} loss:{:.4f} accuracy:{:.4f}'.format(
                (epoch + 1), phase, epoch_loss, epoch_accuracy))
    return model


batch = next(iter(eval_data))

""" Ejecucion """
""" ********* """

epochs = 4000
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

# Carga el modelo RESNET-18.
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_filters = model.fc.in_features

# Se define una capa lineal de dos salidas.
model.fc = nn.Linear(num_filters, 2)
model = model.to(device)

# Se define el optimizador.
optimizer = torch.optim.SGD(model.fc.parameters(),
                            lr=learning_rate, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Se realiza el entrenamiento.
result = train(model, data_loaders, criterion, optimizer, scheduler, epochs)
torch.save(result, f'result/cough_result.model')
