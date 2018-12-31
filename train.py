from torchvision import datasets, transforms, models
import torch, os, json, argparse
from torch import nn, optim
from collections import OrderedDict


def main():
    input_arguments = input_argparse()
    device = device_in_use(gpu_ind=input_arguments.gpu)
    model_mode = ['train', 'valid', 'test']
    model_name = input_arguments.arch
    model = build_model(model_name=model_name, hidden_units=input_arguments.hidden_units)
    train_model(data_dir=input_arguments.data_dir, model=model, device=device, model_mode=model_mode,
                learning_rate=input_arguments.learning_rate, model_name=model_name,
                checkpoint_loc=input_arguments.save_dir, hidden_units=input_arguments.hidden_units)


def build_model(model_name='vgg16', hidden_units=1024):
    model = getattr(models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU(inplace=True)),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model


def save_checkpoint(model, optimizer, epochs, image_input, learning_rate, model_name, hidden_units, checkpoint_loc):
    model.class_to_idx = image_input.class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'hidden_units': hidden_units,
                  'optimizer_state': optimizer.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'model_name': model_name,
                  'learning_rate': learning_rate,
                  'classifier': model.classifier,
                  }
    torch.save(checkpoint, checkpoint_loc)


def load_checkpoint(checkpoint_loc='checkpoint.pth'):
    checkpoint = torch.load(checkpoint_loc)
    model_name = checkpoint['model_name']
    model = build_model(model_name=model_name, hidden_units=checkpoint['hidden_units'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def do_evaluate(device, model, inputs):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in inputs:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network over the %d test images: %d %%' % (total, (100 * correct / total)))


def do_training(model, trainloader, validationloader, criterion, optimizer, device, print_every=10, steps=0, epochs=3):
    running_loss = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        if e % 2 == 0:
            loader = validationloader
            model.eval()
            accuracy = 0
            for ii, (inputs, labels) in enumerate(loader):
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                val_loss = criterion(outputs, labels)
                ps = torch.exp(outputs).data
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()
                if steps % print_every == 0:
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Validation Loss: {:.3f}.. ".format(val_loss / len(validationloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy / len(validationloader)))

        else:
            model.train()
            loader = trainloader
            for ii, (inputs, labels) in enumerate(loader):
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                val_loss = criterion(outputs, labels)
                val_loss.backward()
                optimizer.step()
                running_loss += val_loss.item()
                if steps % print_every == 0:
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Validation Loss: {:.3f}.. ".format(val_loss / len(validationloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy / len(validationloader)))
                    running_loss = 0
    return model


def train_model(device, model, model_mode, model_name, hidden_units, checkpoint_loc, data_dir='flowers/', step=0,
                epochs=3, print_every=10, learning_rate=0.0005):
    '''Define data transforms, image datasets and data loaders'''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in model_mode
                      }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=32)
                   for x in model_mode
                   }

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model = do_training(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, device,
                        print_every=print_every, steps=step, epochs=epochs)
    do_evaluate(device=device, model=model, inputs=dataloaders[model_mode[1]])
    print("Saving the model...")
    save_checkpoint(model=model, optimizer=optimizer, epochs=epochs, hidden_units=hidden_units,
                    checkpoint_loc=checkpoint_loc, image_input=image_datasets[model_mode[2]],
                    learning_rate=learning_rate, model_name=model_name)


def device_in_use(gpu_ind=True):
    if gpu_ind and torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def input_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='flowers/',
                        help='set the directory where the data is present')
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth',
                        help='directory where the checkpoint will be saved')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='select the pretrained model')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='set the training model learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='set the training model\'s hidden units')
    parser.add_argument('--epochs', type=int, default=3,
                        help='set the training model epoch')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable cuda')
    return parser.parse_args()


if __name__ == '__main__':
    main()
