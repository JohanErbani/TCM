import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from useful_functions import *
from confusion_matrices import *
from torchvision.io import read_image, ImageReadMode
import ast
import os
from torchvision import models, transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform):
        self.label_name = 'hard_label'
        self.img_labels = pd.read_csv(annotations_file)
        not_nan = self.img_labels['image_id'] != 0
        self.img_labels = self.img_labels[not_nan].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_nb = str(self.img_labels['image_id'].iloc[idx])
        zeros2add = 12 - len(image_nb)
        image_name = '0' * zeros2add + image_nb + '.jpg'
        img_path = os.path.join(self.img_dir, image_name)
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = torch.tensor(ast.literal_eval(self.img_labels[self.label_name].iloc[idx]), dtype=torch.float32)
        image = self.transform(image)
        return image, label


class dataset(Dataset):
    def __init__(self, dfx, dfy):
        self.x = dfx
        self.y = dfy

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x.iloc[idx], torch.tensor(self.y.iloc[idx], dtype=torch.float32)


def tokenizer_preprocessing(text, tokenizer, max_length=300):
    return tokenizer(text, padding='max_length', max_length=max_length, truncation=True,
                     return_tensors='pt')

class fine_tuning_transformer(nn.Module):
    def __init__(self, transformer, C):
        super().__init__()
        self.transformer = transformer
        self.fine_tuning = nn.Sequential(
            nn.Dropout(.1),
            nn.LazyLinear(C))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        mask = input['attention_mask'].squeeze()
        input_id = input['input_ids'].squeeze()
        outputs = self.transformer(input_id, attention_mask=mask, return_dict=True).last_hidden_state[:, 0, :]
        output = self.fine_tuning(outputs)

        if not self.training:
            return self.sigmoid(output)
        else:
            return output


class images_model(nn.Module):
    def __init__(self, C, hard=True):
        super().__init__()
        self.model = models.regnet_y_400mf(weights='IMAGENET1K_V1')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, C)

        if hard:
            self.activation_function = nn.Sigmoid()
        else:
            self.activation_function = nn.ReLU()

    def forward(self, input):
        output = self.model(input)
        if not self.training:
            return self.activation_function(output)
        else:
            return output

class MyCustomLoss(nn.Module):
    def __init__(self, weights, device):
        super(MyCustomLoss, self).__init__()
        self.weights = weights.to(device)
        self.relu = nn.ReLU()

    def forward(self, input, target):
        positives = torch.zeros_like(target)
        positives[target>0]=1.
        loss = (self.relu(input) - target).pow(2) * positives*self.weights
        return loss.mean()

    def __call__(self, input, target):
        return self.forward(input, target)


def train(train_loader, model, criterion, optim, epoch, device):
    epoch_loss = 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        print('\rEpoch:{}... {:.0f}%'.format(epoch, 100. * i / len(train_loader)), end='')
        x = x.to(device)
        y = y.to(device)

        output = model.forward(x)

        loss = criterion(output, y)
        loss.backward()

        epoch_loss += loss.cpu().detach().numpy()

        optim.step()
        model.zero_grad()

    return epoch_loss


def test(loader, criterion, model, device, epoch, C, f1s_weighted, losses, path, task, weigh):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0
        instances = collect_instances(C)
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            output = model.forward(x)

            loss = criterion(output, y)
            epoch_loss += loss.cpu().detach().numpy()

            instances.update(y, output)

        labels = instances.get_labels()
        predictions = instances.get_predictions()

        M = confusion_matrix(C, type='TCMone')
        M.update(labels, predictions)

        std_binary_predictions = hard(predictions, [.5 for _ in range(C)])
        f1, f1_micro, f1_macro, f1_weighted = all_metrics(labels, std_binary_predictions)

        if epoch > 0 and f1_weighted > max(f1s_weighted):
            torch.save(model.state_dict(), path)

        f1, f1_micro, f1_macro, f1_weighted = all_metrics(labels, hard(predictions, [.5 for _ in range(C)]))

        M = M.get(normalisation='true')

        print('LOSS:{:.2f}'.format(epoch_loss))

        return f1, f1_micro, f1_macro, f1_weighted, epoch_loss, M


def how_many_classes_per_instance(instances):
    return int(100 * instances.sum(dim=1).mean().item()) / 100

def simple(x, c):
    return torch.column_stack(
        [x[:, i] + x[:, i + c] + x[:, i + 2*c] + x[:, i + 3*c] + x[:, i + 4*c] + x[:, i + 5*c]
         for i in range(c)])

def comparison(experience_name, test_loader, train_loader, types, model, device, C, class_name):
    with torch.no_grad():
        model.eval()
        instances = collect_instances(C)
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            output = model.forward(x)

            instances.update(y, output)

        thresholds = instances.get_thresholds()

        instances = collect_instances(C)
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            output = model.forward(x)

            instances.update(y, output)

        labels = instances.get_labels()
        predictions = instances.get_predictions()

        std_binary_predictions = hard(predictions, [.5 for _ in range(C)])
        binary_predictions = hard(predictions, thresholds)

        print('labels:', how_many_classes_per_instance(labels))
        print('opt:', how_many_classes_per_instance(binary_predictions))

        all_metrics(labels, binary_predictions)
        all_metrics(labels, std_binary_predictions)

        Ms = {type: confusion_matrix(C, type=type) for type in types}
        if C!=36 and C!=30:
            Ms = {type: confusion_matrix(C, type=type) for type in types}

        else:
            Ms = {type: confusion_matrix(int(C/6), type=type) for type in types}
            # binary_predictions = predictions
            labels = simple(instances.get_labels(), int(C/6))
            binary_predictions = simple(binary_predictions, int(C/6))


        for k, v in Ms.items():
            v.update(labels, binary_predictions)

        print('\n')
        for k, v in Ms.items():
            print(k)
            torch.save(v.get(), 'Matrix/' + k + '_' + experience_name + '.pt')

            v.print(normalization='raw', class_name = class_name, diag=True)
            print_list(v.score().tolist())

        print('\n')
