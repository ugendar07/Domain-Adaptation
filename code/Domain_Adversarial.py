
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
from skimage import io
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from tqdm import tqdm,trange
from torch.nn.functional import mse_loss
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as f
import glob
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50,ResNet50_Weights
from ResNet50 import *


from torch.autograd import Function


class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, lmbda):
        ctx.lmbda = lmbda

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lmbda

        return output, None



transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224),antialias=True),

        ]
    )


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained_ResNet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
pretrained_ResNet50 = pretrained_ResNet50.to(DEVICE)


# DANN architecture implementation at first stage of base ResNet50 classifier


class DANN_1(nn.Module):

    def __init__(self, model_layers, num_classes):
        super(DANN_1, self).__init__()
        self.feature_extractor = nn.Sequential(*model_layers)

        self.label_predictor = nn.Sequential(
            nn.Linear(2048*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,num_classes),
            nn.Softmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(2048*7*7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x, lmbda):
        x = self.feature_extractor(x)
        x = x.view(-1, 2048*7*7)
        x_reverse = GradientReversal.apply(x, lmbda)
        label_output = self.label_predictor(x)
        domain_output = self.domain_classifier(x_reverse)
        return label_output, domain_output


# Domain-Adversarial Training of Neural Networks (DANN) on the OfficeHome dataset


def DANN_Training_OfficeHome(DEVICE:str ,BATCH_SIZE:int ,NUM_EPOCHS:int , LEARNING_RATE: float)-> None:


    os.makedirs('./snapshots_2', exist_ok=True)

    SOURCE_IMAGE_DIR = './OfficeHomeDataset/Real World/*'
    SOURCE_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(SOURCE_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            SOURCE_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    source_ds = Data(SOURCE_IMAGE_PATH, labels, classes=65, transform=transformations)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True)


    TARGET_IMAGE_DIR = './OfficeHomeDataset/Clipart/*'
    TARGET_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(TARGET_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            TARGET_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    target_ds = Data(TARGET_IMAGE_PATH, labels, classes=65, transform=transformations)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True)

    NUM_CHANNELS = 3 # depends on the dataset
    NUM_CLASSES = 65

    model = pretrained_ResNet50
    model_layers = list(model.children())[:-2]

    model = DANN_1(model_layers, NUM_CLASSES).to(DEVICE)
    label_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    len_dataloader = min(len(source_loader), len(target_loader))
    writer = SummaryWriter('./runs/OfficeHome2_S1')

    iter = 1

    for epoch in trange(NUM_EPOCHS):
        data_zip = enumerate(zip(source_loader, target_loader))
        for batch_idx, ((source_image,source_labels), (target_image,target_labels)) in data_zip:

            all_lbls = []
            all_preds = []

            p = float(batch_idx + epoch * len_dataloader) / 100 / len_dataloader
            lmbda = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = source_image.to(DEVICE)
            target_image = target_image.to(DEVICE)
            source_labels = source_labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            label_output, domain_output = model(source_image, lmbda)
            source_error_label = label_loss(label_output, source_labels)
            source_error_domain = domain_loss(domain_output, torch.ones_like(domain_output))

            domain_lbls = torch.ones_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            _, domain_output = model(target_image, lmbda)
            target_error_domain = domain_loss(domain_output, torch.zeros_like(domain_output))

            domain_lbls = torch.zeros_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            all_lbls = torch.tensor(all_lbls)
            all_preds = torch.tensor(all_preds)
            acc = (all_preds == all_lbls).float().mean()

            error = source_error_label + source_error_domain + target_error_domain

            model.zero_grad()
            error.backward()
            optimizer.step()


            with torch.no_grad():
                writer.add_scalar('training loss', error, global_step = iter-1)
                writer.add_scalar('domain classifier accuracy', acc, global_step = iter-1)

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(target_loader)} \
                    Loss: {error:.4f}  Domain Acc: {acc:.4f}"
                    )

            if(iter % 1000 == 0):
                abc = iter
                torch.save(model.state_dict(), os.path.join(f'./snapshots_2',f'OfficeHomeS1_{abc}.pth'))


            iter += 1




def DANN_Testing_Clipart(DEVICE: str,BATCH_SIZE:int)-> None:

    TEST_IMAGE_DIR = './OfficeHomeDataset/Clipart/*'
    TEST_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(TEST_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            TEST_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    test_ds = Data(TEST_IMAGE_PATH, labels, classes=65, transform=transformations)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    NUM_CHANNELS = 3 # depends on the dataset
    NUM_CLASSES = 65


    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-3]


    NUM_CHANNELS = 3 # depends on the dataset
    NUM_CLASSES = 65
    model = DANN_1(model_layers, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('./snapshots_2/OfficeHomeS1_8000.pth', map_location=torch.device('cpu')))
    model.eval()

    all_lbls = []
    all_preds = []
    for batch_idx, (image,label) in enumerate(test_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        prediction = model(image) 

        ret, lbls = torch.max(label, 1)
        ret, preds = torch.max(prediction, 1)

        lbls = lbls.tolist()
        preds = preds.tolist()

        all_lbls.extend(lbls)
        all_preds.extend(preds)

    all_lbls = torch.tensor(all_lbls)
    all_preds = torch.tensor(all_preds)

    acc = (all_preds == all_lbls).float().mean()
    print(f'The test accuracy on Clipart domain is: {acc}')






# Domain-Adversarial Training of Neural Networks (DANN) on the MNIST dataset


def DANN_Training_MNIST(DEVICE:str ,BATCH_SIZE:int,NUM_EPOCHS:int ,LEARNING_RATE:float )-> None:


    import h5py
    with h5py.File('./USPS/usps.h5', 'r') as hf:
            train = hf.get('train')
            X = train.get('data')[:]
            y = train.get('target')[:]


    os.makedirs('./snapshots_2', exist_ok=True)
    NUM_CHANNELS = 1 # depends on the dataset
    NUM_CLASSES = 10

    source_ds = torchvision.datasets.MNIST('./MNIST', transform=transformations, train=True, download=True)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True)

    y = list(y)
    target_ds = USPS_Data(X, y, classes=NUM_CLASSES, transform=transformations)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-3]

    model = DANN_1(model_layers, NUM_CLASSES).to(DEVICE)
    label_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    len_dataloader = min(len(source_loader), len(target_loader))
    writer = SummaryWriter('./runs/MNIST2_S1')

    iter = 1

    for epoch in trange(NUM_EPOCHS):
        data_zip = enumerate(zip(source_loader, target_loader))
        for batch_idx, ((source_image,source_labels), (target_image,target_labels)) in data_zip:

            all_lbls = []
            all_preds = []

            p = float(batch_idx + epoch * len_dataloader) / 100 / len_dataloader
            lmbda = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = source_image.to(DEVICE)
            target_image = target_image.to(DEVICE)
            source_labels = source_labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            label_output, domain_output = model(source_image, lmbda)
            source_error_label = label_loss(label_output, source_labels)
            source_error_domain = domain_loss(domain_output, torch.ones_like(domain_output))

            domain_lbls = torch.ones_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            _, domain_output = model(target_image, lmbda)
            target_error_domain = domain_loss(domain_output, torch.zeros_like(domain_output))

            domain_lbls = torch.zeros_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            all_lbls = torch.tensor(all_lbls)
            all_preds = torch.tensor(all_preds)
            acc = (all_preds == all_lbls).float().mean()

            error = source_error_label + source_error_domain + target_error_domain

            model.zero_grad()
            error.backward()
            optimizer.step()


            with torch.no_grad():
                writer.add_scalar('training loss', error, global_step = iter-1)
                writer.add_scalar('domain classifier accuracy', acc, global_step = iter-1)

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(target_loader)} \
                    Loss: {error:.4f}  Domain Acc: {acc:.4f}"
                )

            if(iter % 1000 == 0):
                abc = iter
                torch.save(model.state_dict(), os.path.join(f'./snapshots_2',f'MNISTS1_{abc}.pth'))


            iter += 1




# Domain-Adversarial Testing of Neural Networks (DANN) on the USPS dataset

def DANN_Testing_USPS(DEVICE:str, BATCH_SIZE:int )-> None:

    import h5py
    with h5py.File('./USPS/usps.h5', 'r') as hf:

        train = hf.get('train')
        X = train.get('data')[:]
        y = train.get('target')[:]


    NUM_CHANNELS = 1 # depends on the dataset
    NUM_CLASSES = 10

    y = list(y)
    test_ds = USPS_Data(X, y, classes=NUM_CLASSES, transform=transformations)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-3]

    model = DANN_1(model_layers, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('./snapshots_2/MNISTS1_5000.pth', map_location=DEVICE))
    model.eval()

    all_lbls = []
    all_preds = []
    len_dataloader = len(test_loader)
    for batch_idx, (image,label) in enumerate(test_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        p = float(batch_idx + 1 * len_dataloader) / 100 / len_dataloader
        lmbda = 2. / (1. + np.exp(-10 * p)) - 1

        prediction, _ = model(image, lmbda)

        ret, lbls = torch.max(label, 1)
        ret, preds = torch.max(prediction, 1)

        lbls = lbls.tolist()
        preds = preds.tolist()

        all_lbls.extend(lbls)
        all_preds.extend(preds)

    all_lbls = torch.tensor(all_lbls)
    all_preds = torch.tensor(all_preds)

    acc = (all_preds == all_lbls).float().mean()
    print(f'The test accuracy on MNIST-USPS domain is: {acc}')




# DANN architecture implementation at second stage of base ResNet50 classifier

class DANN_2(nn.Module):
    def __init__(self, model_layers, num_classes):
        super(DANN_2, self).__init__()
        self.feature_extractor = nn.Sequential(*model_layers)

        self.label_predictor = nn.Sequential(
            nn.Linear(1024*14*14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,num_classes),
            nn.Softmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(1024*14*14, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, lmbda):
        x = self.feature_extractor(x)
        x = x.view(-1,1024*14*14)
        x_reverse = GradientReversal.apply(x, lmbda)
        label_output = self.label_predictor(x)
        domain_output = self.domain_classifier(x_reverse)
        return label_output, domain_output



# Domain-Adversarial Training of Neural Networks (DANN) on the OfficeHome dataset


def DANN2_Training_OfficeHome(DEVICE:str, BATCH_SIZE:int ,NUM_EPOCHS:int ,LEARNING_RATE:float)-> None:

    os.makedirs('./snapshots_2', exist_ok=True)

    SOURCE_IMAGE_DIR = './OfficeHomeDataset/Real World/*'
    SOURCE_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(SOURCE_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            SOURCE_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    source_ds = Data(SOURCE_IMAGE_PATH, labels, classes=65, transform=transformations)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True)


    TARGET_IMAGE_DIR = './OfficeHomeDataset/Clipart/*'
    TARGET_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(TARGET_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            TARGET_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    target_ds = Data(TARGET_IMAGE_PATH, labels, classes=65, transform=transformations)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True)

    NUM_CHANNELS = 3 # depends on the dataset
    NUM_CLASSES = 65
 
    model = pretrained_ResNet50
    model_layers = list(model.children())[:-3]

    model = DANN_2(model_layers, NUM_CLASSES).to(DEVICE)
    label_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    len_dataloader = min(len(source_loader), len(target_loader))
    writer = SummaryWriter('./runs/OfficeHome2_S2')

    iter = 1
    for epoch in trange(NUM_EPOCHS):
        data_zip = enumerate(zip(source_loader, target_loader))
        for batch_idx, ((source_image,source_labels), (target_image,target_labels)) in data_zip:

            all_lbls = []
            all_preds = []

            p = float(batch_idx + epoch * len_dataloader) / 100 / len_dataloader
            lmbda = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = source_image.to(DEVICE)
            target_image = target_image.to(DEVICE)
            source_labels = source_labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            label_output, domain_output = model(source_image, lmbda)
            source_error_label = label_loss(label_output, source_labels)
            source_error_domain = domain_loss(domain_output, torch.ones_like(domain_output))

            domain_lbls = torch.ones_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            _, domain_output = model(target_image, lmbda)
            target_error_domain = domain_loss(domain_output, torch.zeros_like(domain_output))

            domain_lbls = torch.zeros_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            all_lbls = torch.tensor(all_lbls)
            all_preds = torch.tensor(all_preds)
            acc = (all_preds == all_lbls).float().mean()

            error = source_error_label + source_error_domain + target_error_domain
    

            model.zero_grad()
            error.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(target_loader)} \
                    Loss: {error:.4f}  Domain Acc: {acc:.4f}"
                )
            if(iter % 1000 == 0):
                abc = iter
                torch.save(model.state_dict(), os.path.join(f'./snapshots_2',f'OfficeHomeS2_{abc}.pth'))

            iter += 1



# Domain-Adversarial Training of Neural Networks (DANN) on the MNIST dataset


def DANN2_Training_MNIST(DEVICE:str, BATCH_SIZE:int , NUM_EPOCHS:int , LEARNING_RATE:float)-> None:

    import h5py
    with h5py.File('./USPS/usps.h5', 'r') as hf:
        train = hf.get('train')
        X = train.get('data')[:]
        y = train.get('target')[:]



    os.makedirs('./snapshots_2', exist_ok=True)
    NUM_CHANNELS = 1 # depends on the dataset
    NUM_CLASSES = 10

    source_ds = torchvision.datasets.MNIST('./MNIST', transform=transformations, train=True, download=True)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True)

    y = list(y)
    target_ds = USPS_Data(X, y, classes=NUM_CLASSES, transform=transformations)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-4]

    model = DANN_2(model_layers, NUM_CLASSES).to(DEVICE)
    label_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    len_dataloader = min(len(source_loader), len(target_loader))
    writer = SummaryWriter('./runs/MNIST2_S2')

    iter = 1

    for epoch in trange(NUM_EPOCHS):
        data_zip = enumerate(zip(source_loader, target_loader))
        for batch_idx, ((source_image,source_labels), (target_image,target_labels)) in data_zip:

            all_lbls = []
            all_preds = []

            p = float(batch_idx + epoch * len_dataloader) / 100 / len_dataloader
            lmbda = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = source_image.to(DEVICE)
            target_image = target_image.to(DEVICE)
            source_labels = source_labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            label_output, domain_output = model(source_image, lmbda)
            source_error_label = label_loss(label_output, source_labels)
            source_error_domain = domain_loss(domain_output, torch.ones_like(domain_output))

            domain_lbls = torch.ones_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            _, domain_output = model(target_image, lmbda)
            target_error_domain = domain_loss(domain_output, torch.zeros_like(domain_output))

            domain_lbls = torch.zeros_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            all_lbls = torch.tensor(all_lbls)
            all_preds = torch.tensor(all_preds)
            acc = (all_preds == all_lbls).float().mean()

            error = source_error_label + source_error_domain + target_error_domain

            model.zero_grad()
            error.backward()
            optimizer.step()


            with torch.no_grad():
                writer.add_scalar('training loss', error, global_step = iter-1)
                writer.add_scalar('domain classifier accuracy', acc, global_step = iter-1)

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(target_loader)} \
                    Loss: {error:.4f}  Domain Acc: {acc:.4f}"
                )

            if(iter % 1000 == 0):
                abc = iter
                torch.save(model.state_dict(), os.path.join(f'./snapshots_2',f'MNISTS2_{abc}.pth'))


            iter += 1



# Domain-Adversarial Testing of Neural Networks (DANN) on the USPS dataset


def DANN2_Testing_USPS(DEVICE:str, BATCH_SIZE:int , NUM_EPOCHS:int , LEARNING_RATE:float)-> None:

    NUM_CHANNELS = 1 # depends on the dataset
    NUM_CLASSES = 10

    y = list(y)
    test_ds = USPS_Data(X, y, classes=NUM_CLASSES, transform=transformations)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-4]

    model = DANN_1(model_layers, NUM_CLASSES).to(DEVICE)    
    model.load_state_dict(torch.load('./snapshots_2/MNISTS2_6000.pth', map_location=DEVICE))
    model.eval()

    all_lbls = []
    all_preds = []
    len_dataloader = len(test_loader)
    for batch_idx, (image,label) in enumerate(test_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        p = float(batch_idx + 1 * len_dataloader) / 100 / len_dataloader
        lmbda = 2. / (1. + np.exp(-10 * p)) - 1
  
        prediction, _ = model(image, lmbda) 

        ret, lbls = torch.max(label, 1)
        ret, preds = torch.max(prediction, 1)

        lbls = lbls.tolist()
        preds = preds.tolist()

        all_lbls.extend(lbls)
        all_preds.extend(preds)

    all_lbls = torch.tensor(all_lbls)
    all_preds = torch.tensor(all_preds)

    acc = (all_preds == all_lbls).float().mean()
    print(f'The test accuracy on MNIST-USPS domain is: {acc}')




# DANN architecture implementation at third stage of base ResNet50 classifier


class DANN_3(nn.Module):

    def __init__(self, model_layers, num_classes):
        super(DANN_3, self).__init__()
        self.feature_extractor = nn.Sequential(*model_layers)

        self.label_predictor = nn.Sequential(
            nn.Linear(512*28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512,num_classes),
            nn.Softmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512*28*28, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, lmbda):
        x = self.feature_extractor(x)
        x = x.view(-1,512*28*28)
        x_reverse = GradientReversal.apply(x, lmbda)
        label_output = self.label_predictor(x)
        domain_output = self.domain_classifier(x_reverse)
        return label_output, domain_output




# Domain-Adversarial Training of Neural Networks (DANN) on the OfficeHome dataset

def DANN3_Training_OfficeHome(DEVICE:str, BATCH_SIZE:int, NUM_EPOCHS:int, LEARNING_RATE:float)-> None:

    os.makedirs('./snapshots_2', exist_ok=True)

    SOURCE_IMAGE_DIR = './OfficeHomeDataset/Real World/*'
    SOURCE_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(SOURCE_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            SOURCE_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    source_ds = Data(SOURCE_IMAGE_PATH, labels, classes=65, transform=transformations)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True)


    TARGET_IMAGE_DIR = './OfficeHomeDataset/Clipart/*'
    TARGET_IMAGE_PATH = []
    labels = []
    lbl = 0
    for img_type in sorted(glob.glob(TARGET_IMAGE_DIR)):
        for path in sorted(glob.glob(img_type + "/*")):
            TARGET_IMAGE_PATH.append(path)
            labels.append(lbl)
        lbl += 1
    target_ds = Data(TARGET_IMAGE_PATH, labels, classes=65, transform=transformations)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True)

    NUM_CHANNELS = 3 # depends on the dataset
    NUM_CLASSES = 65

    model = pretrained_ResNet50
    model_layers = list(model.children())[:-4]

    model = DANN_3(model_layers, NUM_CLASSES).to(DEVICE)
    label_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    len_dataloader = min(len(source_loader), len(target_loader))
    writer = SummaryWriter('./runs/OfficeHome2_S3')

    iter = 1
    for epoch in trange(NUM_EPOCHS):
        data_zip = enumerate(zip(source_loader, target_loader))
        for batch_idx, ((source_image,source_labels), (target_image,target_labels)) in data_zip:

            all_lbls = []
            all_preds = []

            p = float(batch_idx + epoch * len_dataloader) / 100 / len_dataloader
            lmbda = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = source_image.to(DEVICE)
            target_image = target_image.to(DEVICE)
            source_labels = source_labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            label_output, domain_output = model(source_image, lmbda)
            source_error_label = label_loss(label_output, source_labels)
            source_error_domain = domain_loss(domain_output, torch.ones_like(domain_output))

            domain_lbls = torch.ones_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            _, domain_output = model(target_image, lmbda)
            target_error_domain = domain_loss(domain_output, torch.zeros_like(domain_output))

            domain_lbls = torch.zeros_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            all_lbls = torch.tensor(all_lbls)
            all_preds = torch.tensor(all_preds)
            acc = (all_preds == all_lbls).float().mean()

            error = source_error_label + source_error_domain + target_error_domain


            model.zero_grad()
            error.backward()
            optimizer.step()

            with torch.no_grad():
                writer.add_scalar('training loss', error, global_step = iter-1)
                writer.add_scalar('domain classifier accuracy', acc, global_step = iter-1)

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(target_loader)} \
                    Loss: {error:.4f}  Domain Acc: {acc:.4f}"
                )
            if(iter % 1000 == 0):
                abc = iter
                torch.save(model.state_dict(), os.path.join(f'./snapshots_2',f'OfficeHomeS3_{abc}.pth'))

            iter += 1





# Domain-Adversarial Training of Neural Networks (DANN) on the MNIST dataset


def DANN3_Training_MNIST(DEVICE:str, BATCH_SIZE:int, NUM_EPOCHS:int, LEARNING_RATE:float)-> None:

    import h5py
    with h5py.File('./USPS/usps.h5', 'r') as hf:
        train = hf.get('train')
        X = train.get('data')[:]
        y = train.get('target')[:]


    os.makedirs('./snapshots_2', exist_ok=True)
    NUM_CHANNELS = 1 # depends on the dataset
    NUM_CLASSES = 10

    source_ds = torchvision.datasets.MNIST('./MNIST', transform=transformations, train=True, download=True)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True)

    y = list(y)
    target_ds = USPS_Data(X, y, classes=NUM_CLASSES, transform=transformations)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-5]

    model = DANN_1(model_layers, NUM_CLASSES).to(DEVICE)
    label_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    len_dataloader = min(len(source_loader), len(target_loader))
    writer = SummaryWriter('./runs/MNIST2_S3')

    iter = 1

    for epoch in trange(NUM_EPOCHS):
        data_zip = enumerate(zip(source_loader, target_loader))
        for batch_idx, ((source_image,source_labels), (target_image,target_labels)) in data_zip:

            all_lbls = []
            all_preds = []

            p = float(batch_idx + epoch * len_dataloader) / 100 / len_dataloader
            lmbda = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = source_image.to(DEVICE)
            target_image = target_image.to(DEVICE)
            source_labels = source_labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            label_output, domain_output = model(source_image, lmbda)
            source_error_label = label_loss(label_output, source_labels)
            source_error_domain = domain_loss(domain_output, torch.ones_like(domain_output))

            domain_lbls = torch.ones_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            _, domain_output = model(target_image, lmbda)
            target_error_domain = domain_loss(domain_output, torch.zeros_like(domain_output))

            domain_lbls = torch.zeros_like(domain_output)
            lbls = domain_lbls.round()
            preds = domain_output.round()
            lbls = lbls.tolist()
            preds = preds.tolist()

            all_lbls.extend(lbls)
            all_preds.extend(preds)

            all_lbls = torch.tensor(all_lbls)
            all_preds = torch.tensor(all_preds)
            acc = (all_preds == all_lbls).float().mean()

            error = source_error_label + source_error_domain + target_error_domain  

            model.zero_grad()
            error.backward()
            optimizer.step()


            with torch.no_grad():
                writer.add_scalar('training loss', error, global_step = iter-1)
                writer.add_scalar('domain classifier accuracy', acc, global_step = iter-1)
    
            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(target_loader)} \
                    Loss: {error:.4f}  Domain Acc: {acc:.4f}"
                )

            if(iter % 1000 == 0):
                abc = iter
                torch.save(model.state_dict(), os.path.join(f'./snapshots_2',f'MNISTS3_{abc}.pth'))


            iter += 1



# Domain-Adversarial Testing of Neural Networks (DANN) on the USPS dataset


def DANN3_Testing_USPS(DEVICE:str, BATCH_SIZE:int, NUM_EPOCHS:int, LEARNING_RATE:float)-> None:

    NUM_CHANNELS = 1 # depends on the dataset
    NUM_CLASSES = 10

    y = list(y)
    test_ds = USPS_Data(X, y, classes=NUM_CLASSES, transform=transformations)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    model_layers = list(model.children())[:-5]

    model = DANN_3(model_layers, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('./snapshots_2/MNISTS3_6000.pth', map_location=DEVICE))
    model.eval()

    all_lbls = []
    all_preds = []
    len_dataloader = len(test_loader)
    for batch_idx, (image,label) in enumerate(test_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        p = float(batch_idx + 1 * len_dataloader) / 100 / len_dataloader
        lmbda = 2. / (1. + np.exp(-10 * p)) - 1

        prediction, _ = model(image, lmbda)

        ret, lbls = torch.max(label, 1)
        ret, preds = torch.max(prediction, 1)

        lbls = lbls.tolist()
        preds = preds.tolist()

        all_lbls.extend(lbls)
        all_preds.extend(preds)

    all_lbls = torch.tensor(all_lbls)
    all_preds = torch.tensor(all_preds)

    acc = (all_preds == all_lbls).float().mean()
    print(f'The test accuracy on MNIST-USPS domain is: {acc}')


