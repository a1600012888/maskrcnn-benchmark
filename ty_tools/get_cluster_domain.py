import torch
import torchvision
from tylib.tytorch.layers.basic import Lambda
import json
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm

Identity = Lambda(lambda x: x)

bdd_val_label_path = '../bdd100k/labels/bdd100k_labels_images_val.json'
bdd_train_label_path = '../bdd100k/labels/bdd100k_labels_images_train.json'

val_dir = '../bdd100k/images/100k/val'
train_dir = '../bdd100k/images/100k/train'


save_dir = '../bdd100k/domain_embedding_scatter'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

name2domain = {}


def get_domain_idx(img,) -> int:

    domain_idx = 0

    return domain_idx


def read_bdd_labels(path):
    with open(path, 'r') as f:
        bdd = json.load(f)
    return bdd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class BddDataset(torch.utils.data.Dataset):

    def __init__(self, ann_path, img_dir, transform, ):
        self.anns = read_bdd_labels(ann_path)
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_name = ann['name']
        img_path = os.path.join(self.img_dir, img_name)
        img = pil_loader(img_path)

        img_tensor = self.transform(img)

        return img_tensor, img_name


def run_one_epoch(net, dl, num=10):
    net.eval()
    all_vecs = []
    all_names = []
    with torch.no_grad():
        pbar = tqdm(dl)
        for i, (data, img_names) in enumerate(pbar):
            data = data.cuda()

            vectors = net(data)
            vectors = vectors.cpu().numpy()

            for vec, name in zip(vectors, img_names):
                name = str(name)
                all_vecs.append(vec)
                all_names.append(name)


    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num, random_state=0).fit(all_vecs)

    return kmeans, all_names

if __name__ == '__main__':

    num = 10
    net = torchvision.models.resnet50(pretrained=True)
    #net.fc = torch.nn.Linear(2048, 200)
    #net.load_state_dict(torch.load('../models/attr-cls-res50.pth')['model'])
    net.fc = Identity

    net.cuda()
    net.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    ds_train = BddDataset(bdd_train_label_path, train_dir, data_transform)
    ds_val = BddDataset(bdd_val_label_path, val_dir, data_transform)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, num_workers=8)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=64, num_workers=8)


    kmeans, all_names = run_one_epoch(net, dl_train, num)
    #kmeans, all_names = run_one_epoch(net, dl_val, num)
    # run_one_epoch(net, dl_train, embedding_dic)

    for i in range(num):

        embed = kmeans.cluster_centers_[i]

        save_path = os.path.join(save_dir, str(i) + '.txt')

        np.savetxt(save_path, embed)


    for domain_idx, name in zip(kmeans.labels_, all_names):
        name2domain[name] = int(domain_idx)
        print(domain_idx, name)

    name2domain_str = json.dumps(name2domain)
    with open(os.path.join(save_dir, 'name2domain.json'), 'w') as f:
        f.write(name2domain_str)



