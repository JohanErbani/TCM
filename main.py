import sys
from transformers import AutoModel, AutoTokenizer
from models import *
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
import pandas as pd
import os
import ast

def experience(task, mode, repository_path):
    assert (task == 'text') or (task == 'hard_images') or (task == 'EXIST')
    assert (mode == 'train') or (mode == 'test')
    """
    ###################################################################
    HYPERPARAMETERS ###################################################
    ###################################################################
    """
    seed = 42
    batch_size = 64 if (task == 'text') or (task == 'EXIST') else 128
    if mode == 'test':
        batch_size = 528
    lr = 1e-5 if (task == 'text') or (task == 'EXIST') else 1e-4
    num_epochs = 75

    matrices = ['MLCM', 'MLCTr', 'MLCTp', 'SCMe', 'SCM_min', 'SCM_max', 'TCMone', 'TCMlab', 'TCMpred']

    device = torch.device('cuda')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    """
    ###################################################################
    TASKS #############################################################
    ###################################################################
    """
    if task == 'EXIST':
        df = pd.read_json('/home2/jleydet/Repository/Data/EXIST 2024 Tweets Dataset/training/EXIST2024_training.json').T
        mask_en = df['lang'] == 'en'
        df = df[mask_en].reset_index(drop=True)
        x_train = df['tweet']
        y_train = df['labels_task3'].apply(EXIST)

        df = pd.read_json('/home2/jleydet/Repository/Data/EXIST 2024 Tweets Dataset/dev/EXIST2024_dev.json').T
        mask_en = df['lang'] == 'en'
        df = df[mask_en].reset_index(drop=True)
        x_test = df['tweet']
        y_test = df['labels_task3'].apply(EXIST)

        class_names = ['-', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE',
                       'MISOGYNY-NON-SEXUAL-VIOLENCE'] * 6

        c = len(class_names)

        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        transformer = AutoModel.from_pretrained("vinai/bertweet-base")

        model = fine_tuning_transformer(transformer, c).to(device)

        x_train = x_train.apply(lambda text: tokenizer_preprocessing(text, tokenizer, 120))
        x_test = x_test.apply(lambda text: tokenizer_preprocessing(text, tokenizer, 120))

        train_set = dataset(x_train, y_train)
        test_set = dataset(x_test, y_test)

    elif task == 'text':
        df = pd.read_csv(repository_path + 'Data/mpst_full_data.csv', engine='python')
        mask = df['split'] == 'test'
        df_train = df[~mask].reset_index(drop=True)
        df_test = df[mask].reset_index(drop=True)
        x_name = 'plot_synopsis'
        y_name = 'tags'
        df, class_names, c = get_data(df_train, x_name, y_name)
        x_train = df[x_name]
        y_train = df[y_name]
        df, _, _ = get_data(df_test, x_name, y_name, class_names)
        x_test = df[x_name]
        y_test = df[y_name]

        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        transformer = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        model = fine_tuning_transformer(transformer, c).to(device)

        x_train = x_train.apply(lambda text: tokenizer_preprocessing(text, tokenizer))
        x_test = x_test.apply(lambda text: tokenizer_preprocessing(text, tokenizer))

        train_set = dataset(x_train, y_train)
        test_set = dataset(x_test, y_test)

    else:
        file_path = '/home2/jleydet/Repository/Data/coco_hard_class_names.txt'

        with open(file_path, 'r') as file:
            content = file.read()
        class_names = ast.literal_eval(content)
        c = len(class_names)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_image_dir = '/home2/jleydet/Repository/Data/MS_COCO_2017/train2017'
        train_label_dir = '/home2/jleydet/Repository/Data/train_ms_coco_2017.csv'

        val_image_dir = '/home2/jleydet/Repository/Data/MS_COCO_2017/val2017'
        val_label_dir = '/home2/jleydet/Repository/Data/val_ms_coco_2017.csv'

        train_set = CustomImageDataset(train_label_dir, train_image_dir, preprocess)
        test_set = CustomImageDataset(val_label_dir, val_image_dir, preprocess)
        y_train = pd.read_csv(train_label_dir)['hard_label'].apply(ast.literal_eval)

        model = images_model(c).to(device)

    """
    ###################################################################
    MODELE ############################################################
    ###################################################################
    """
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

    n_positive = torch.tensor(list(y_train)).sum(dim=0)
    n_negative = len(y_train) - n_positive
    pos_weight = n_negative / n_positive

    max_weight = 200 if task == 'text' else 100
    mask_max = pos_weight > max_weight
    pos_weight[mask_max] = max_weight

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loss = []
    test_loss = []
    f1s_micro = []
    f1s_macro = []
    f1s_weighted = []

    experience_name = task

    path = repository_path + 'Models/' + experience_name + '.pth'
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    if mode == 'test':
        comparison(experience_name, test_loader, train_loader, matrices, model, device, c,
                   class_names)
    else:
        for epoch in range(num_epochs):
            train_epoch_loss = train(train_loader, model, criterion, optim, epoch, device)

            train_loss.append(train_epoch_loss)

            f1, f1_micro, f1_macro, f1_weighted, test_epoch_loss, matrice = test(test_loader, criterion, model, device,
                                                                                 epoch,
                                                                                 c, f1s_weighted, test_loss, path, task,
                                                                                 pos_weight)

            f1s_micro.append(f1_micro)
            f1s_macro.append(f1_macro)
            f1s_weighted.append(f1_weighted)
            test_loss.append(test_epoch_loss)

            plot_test(train_loss, test_loss, f1s_micro, f1s_macro, f1s_weighted, matrice)

    save_res(train_loss, test_loss, f1s_micro, f1s_macro, f1s_weighted, experience_name)

repository_path = "/home2/jleydet/Repository/"
for mode in ['train', 'test']:
    for task in ['EXIST','text', 'hard_images']:
        experience(task, mode, repository_path)
