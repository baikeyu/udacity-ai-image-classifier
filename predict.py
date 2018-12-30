from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch, json, argparse, train, os
import numpy as np


def main():
    input_arguments = input_argparse()
    im = input_arguments.input_image_path
    device = train.device_in_use(gpu_ind=input_arguments.gpu)
    cat_to_name = cat_to_name_func(input_arguments.category_names)
    model = train.load_checkpoint(checkpoint_loc=input_arguments.checkpoint_name + '.pth')
    top_probabilities, top_classes = predict(image_path=im, model=model, topk=input_arguments.top_k, device=device)
    top_labels = []
    for i in top_classes:
        top_labels += [cat_to_name[i]]

    print('Predicted flower name: ' + str(cat_to_name[top_classes[0]]) + "\n")
    print('PROBABILITY' + ' ' + 'PREDICTION')
    for probability, prediction in zip(top_probabilities, top_classes):
        print(str(probability) + ' : ' + str(cat_to_name[prediction]))
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    show_result_image(image=im, probability=top_probabilities, top_classes=top_classes, top_labels=top_labels,
                       data_dir=input_arguments.input_image_path)


def show_result_image(image, probability, top_classes, top_labels,  data_dir):
    # Plotting test image and predicted probabilites
    f, ax = plt.subplots(2, figsize=(6, 8))

    im = Image.open(image)
    ax[0].imshow(im)
    ax[0].axis('off')
    ax[0].set_title(top_labels[0])

    y_names = np.arange(len(top_labels))
    ax[1].barh(y_names, probability, color='darkblue')
    ax[1].set_yticks(y_names)
    ax[1].set_yticklabels(top_labels)
    ax[1].invert_yaxis()
    plt.savefig(data_dir.split('/')[2] + '_' + top_classes[0] + '.png')


def cat_to_name_func(name='cat_to_name.json'):
    with open(name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def process_image(image_path):
    pil_img = Image.open(image_path)
    img = transforms.Resize(256)(pil_img)
    lmargin = (img.width - 224) / 2  # left
    bmargin = (img.height - 224) / 2  # bottom
    rmargin = lmargin + 224  # right
    tmargin = bmargin + 224  # top
    img = img.crop((lmargin, bmargin, rmargin, tmargin))
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).type(torch.FloatTensor)


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


def predict(image_path, model, device='cpu', topk=5):

    print(device)
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img.to(device)
    img = img.unsqueeze_(0)

    with torch.no_grad():
        probabilities = torch.exp(model.forward(img))

    topk_prob, topk_idx = probabilities.topk(topk)
    topk_prob = topk_prob.tolist()[0]
    topk_idx = topk_idx.tolist()[0]

    idx_to_class = {y: x for x, y in model.class_to_idx.items()}

    topk_class = []
    for index in topk_idx:
        topk_class += [idx_to_class[index]]

    return topk_prob, topk_class


def input_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path', type=str, default='flowers/test/10/image_07090.jpg',
                        help='set the directory where the image is present')
    parser.add_argument('checkpoint_name', type=str, default='checkpoint',
                        help='checkpoint name')
    parser.add_argument('--top_k', type=int, default=5,
                        help='set the number of top matching results')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='label to name json filename')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable cuda')
    return parser.parse_args()


if __name__ == '__main__':
    main()