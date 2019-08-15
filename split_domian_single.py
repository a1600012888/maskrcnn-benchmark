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

bdd_val_label_path = '/rscratch/data/bdd100k/labels/bdd100k_labels_images_val.json'
bdd_train_label_path = '/rscratch/data/bdd100k/labels/bdd100k_labels_images_train.json'

val_dir = '/rscratch/data/bdd100k/images/100k/val'
train_dir = '/rscratch/data/bdd100k/images/100k/train'

Weathers = ['clear', 'partly cloudy', 'overcast', 'rainy', 'snowy', 'foggy', 'undefined']
Scenes = ['residential', 'highway', 'city street', 'parking lot', 'gas stations', 'tunnel', 'undefined']
TimeofDays = ['dawn/dusk', 'daytime', 'night', 'undefined']


save_dir = '/rscratch/data/bdd100k/domain_embedding_train_single'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

name2domain = {}

world_rank = -1
def get_domain_idx(attributes:dict, Domains) -> int:

    weather, scene, timeofday = attributes['weather'], attributes['scene'], attributes['timeofday']

    w, s, t = Weathers.index(weather), Scenes.index(scene), TimeofDays.index(timeofday)

    global world_rank
    world_rank = world_rank + 1
    print(world_rank)
    return world_rank


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

    def __init__(self, ann_path, img_dir, transform, Domains):
        self.anns = read_bdd_labels(ann_path)
        self.img_dir = img_dir
        self.transform = transform
        self.Domains = Domains

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_name = ann['name']
        img_path = os.path.join(self.img_dir, img_name)
        img = pil_loader(img_path)

        img_tensor = self.transform(img)
        domain_idx = get_domain_idx(ann['attributes'], self.Domains)

        #name2domain[img_name] = domain_idx
        return img_tensor, domain_idx, img_name

def run_one_epoch(net, dl, embedding_dic, Domains):
    with torch.no_grad():
        pbar = tqdm(dl)
        for i, (data, domain_idxs, img_names) in enumerate(pbar):
            data = data.cuda()
            vectors = net(data)
            vectors = vectors.cpu()
            for vec, domain_idx, name in zip(vectors, domain_idxs, img_names):
                name = str(name)
                #print(name)
                domain_idx = int(domain_idx)
                Domains[domain_idx]['num'] += 1
                name2domain[name] = domain_idx
                embedding_dic[domain_idx] = embedding_dic[domain_idx] + vec



if __name__ == '__main__':
    Domains = []
    for i in range(70001):
        Domains.append({'name': i, 'num': 0})

    #for w in Weathers:
    #    for s in Scenes:
    #        for t in TimeofDays:
    #            Domains.append({'name': w + s + t, 'num': 0})

    net = torchvision.models.resnet50(pretrained=True)
    net.fc = Identity
    net.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    ds_train = BddDataset(bdd_train_label_path, train_dir, data_transform, Domains)
    ds_val = BddDataset(bdd_val_label_path, val_dir, data_transform, Domains)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, num_workers=1)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=64, num_workers=1)

    embedding_dic = {i:0 for i in range(70001)}

    run_one_epoch(net, dl_train, embedding_dic, Domains)
    # run_one_epoch(net, dl_train, embedding_dic)
    

    for i in range(len(Domains)):
        dic = Domains[i]
        #print(dic['num'])
        num = dic['num'] * 1.0
        if num != 0.0:

            embedding_dic[i] = embedding_dic[i] / num
            embedding_dic[i] = embedding_dic[i].numpy()
            domain_idx = i
            save_path = os.path.join(save_dir, str(domain_idx) + '.txt')
            np.savetxt(save_path, embedding_dic[i])
        #print(num, embedding_dic[i])

    # for domain_idx, vec in embedding_dic.items():
    #   save_path = os.path.join(save_dir, str(domain_idx)+'.txt')
    #   np.savetxt(save_path, vec)

    Domians_str = json.dumps(Domains)
    with open(os.path.join(save_dir, 'info.json'), 'w') as f:
        f.write(Domians_str)

    name2domain_str = json.dumps(name2domain)
    with open(os.path.join(save_dir, 'name2domain.json'), 'w') as f:
        f.write(name2domain_str)



