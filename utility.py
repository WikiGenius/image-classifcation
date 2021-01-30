import numpy as np
import helper
from PIL import Image
from torchvision import datasets, transforms, models
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import time
import json
import seaborn as sns


def get_train_valid_test_loader(data_dir, gpu):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # some preperation variables
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)

    # train transform
    train_transforms = transforms.Compose([
        transforms.RandomOrder([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224)]),
        transforms.ToTensor(),
        normalize])
    # valid and test transform
    test_valid_transforms = transforms.Compose(
        [resize, crop, transforms.ToTensor(), normalize])

    # TODO: Load the datasets with ImageFolder
    trainsets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validsets = datasets.ImageFolder(
        valid_dir, transform=test_valid_transforms)
    testsets = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    cond_cuda = True if torch.cuda.is_available() and gpu else False
    if cond_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'loading data using device:{device}')
    trainloaders = torch.utils.data.DataLoader(
        trainsets, batch_size=32, shuffle=True, pin_memory=cond_cuda)
    validloaders = torch.utils.data.DataLoader(
        validsets, batch_size=64, shuffle=True, pin_memory=cond_cuda)
    testloaders = torch.utils.data.DataLoader(
        testsets, batch_size=64, shuffle=True, pin_memory=cond_cuda)

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'train': train_transforms,
                       'valid': test_valid_transforms, 'test': test_valid_transforms}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train': trainsets, 'valid': validsets, 'test': testsets}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': trainloaders,
                   'valid': validloaders, 'test': testloaders}

    return dataloaders, image_datasets


def load_checkpoint(path_checkpoint, gpu,  train=True):
    checkpoint = torch.load(path_checkpoint)
    if checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('needs this pretrained model ....', checkpoint['arch'])
        return
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']
    dropout_p = checkpoint['dropout_p']
    classifier_state_dict = checkpoint['state_dict']
    helper.get_classifier(model, hidden_sizes,
                          output_size, dropout_p)

    model.load_state_dict(classifier_state_dict)
    model.class_to_idx = checkpoint['class_to_idx']
    print("Info about checkpoint model:")
    print("#Epochs: {} .. train_loss: {:.3f}, valid_loss: {:.3f}, valid_accuracy: {:.3f}".format(
        checkpoint['epochs'], checkpoint['train_loss'], checkpoint['valid_loss'], checkpoint['valid_accuracy']*100))

    if not train:
        return model
    optimizer = optim.Adam(model.classifier.parameters())
    # load the previus optimizer
    optimizer.load_state_dict(checkpoint['optimzier_state_dict'])
    cond_cuda = True if torch.cuda.is_available() and gpu else False
    if cond_cuda:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    return model, optimizer, hidden_sizes


def save_checkpoint(path_checkpoint, model, hidden_sizes, trainsets, optimizer, criterion, measurements, arch, epochs):
    model.class_to_idx = trainsets.class_to_idx

    checkpoint = {}
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['input_size'] = model.classifier.state_dict()[
        'layers.0.weight'].shape[1]
    checkpoint['output_size'] = len(trainsets.classes)
    checkpoint['hidden_sizes'] = hidden_sizes
    checkpoint['dropout_p'] = 0.2
    checkpoint['optimzier_state_dict'] = optimizer.state_dict()
    checkpoint['class_to_idx'] = trainsets.class_to_idx
    checkpoint['epochs'] = epochs
    checkpoint['train_loss'] = measurements[0]
    checkpoint['valid_loss'] = measurements[1]
    checkpoint['valid_accuracy'] = measurements[2]
    checkpoint['arch'] = arch
    # checkpoint;
    torch.save(checkpoint, path_checkpoint)


def CenterCrop(image, new_width=224):
    new_height = new_width
    width, height = image.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    return image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns numpy
    '''
    # TODO: Process a PIL image for use in a PyTorch model

    # resize the image with  keeping the aspect ratio

    w, h = image.size
    aspect_ratio = w/h
    if w < h:
        image = image.resize((256, int(round(256/aspect_ratio))))
    else:
        image = image.resize((int(round(256*aspect_ratio)), 256))
    # crop out the center 224x224 portion of the image
    w, h = image.size
    image = image.crop(((w-224)/2, (h-224)/2, (w+224)/2, (h+224)/2))
    # the model expected floats 0-1. So, You'll need to convert the values of image
    image = np.array(image)/255

    # Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image[:, :, 0] = (image[:, :, 0] - mean[0])/std[0]
    image[:, :, 1] = (image[:, :, 1] - mean[1])/std[1]
    image[:, :, 2] = (image[:, :, 2] - mean[2])/std[2]
    # reorder dimentions
    image = image.transpose((2, 0, 1))

    return image


def predict(image_path, model, gpu_choice, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    start = time.time()

    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        # in evaluation mode
        model.eval()
        # configure device
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and gpu_choice else "cpu")
        print('predict using ', device)
        model.to(device)
        # loads image
        image = Image.open(image_path)
        image = process_image(image)
        # convert it into tensor
        image = np.array([image], dtype=np.float32)
        image = torch.from_numpy(image)

        image = image.to(device)
        # feedforward
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_probs, top_class = ps.topk(topk, dim=1)
        classes_names = []
        for each in top_class[0]:
            classes_names.extend(
                [class_ for class_, idx in model.class_to_idx.items() if idx == each])
    time_elapsed = (time.time() - start) * 1000
    print("\nTotal time: {:.0f}s {:.0f}ms".format(
        time_elapsed//1000, time_elapsed % 1000))
    return top_probs.view(-1).tolist(), classes_names


def sanityChecking(image_path, cat_to_name_json_path, model, gpu, topk=5):

    with open(cat_to_name_json_path, 'r') as f:
        cat_to_name_json = json.load(f)
    plt.figure(figsize=[18, 5])
    image_path = 'flowers/train/101/image_07945.jpg'
    acutal_class = image_path.split('/')[2]
    actual_name = cat_to_name_json[acutal_class]
    probs, classes = predict(image_path, model, gpu, topk)

    top_names = [cat_to_name_json[class_] for class_ in classes]
    image = Image.open(image_path)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)

    sns.barplot(x=probs, y=top_names, color='blue')
    print(f"pred = {top_names[0]} , actual = {actual_name}")
    plt.show()
    return top_names, probs
