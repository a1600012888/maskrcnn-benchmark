from sklearn.cluster import KMeans
import numpy as np
import os
import json

bdd_val_label_path = '/rscratch/data/bdd100k/labels/bdd100k_labels_images_val.json'
bdd_train_label_path = '/rscratch/data/bdd100k/labels/bdd100k_labels_images_train.json'


Weathers = ['clear', 'partly cloudy', 'overcast', 'rainy', 'snowy', 'foggy', 'undefined']
Scenes = ['residential', 'highway', 'city street', 'parking lot', 'gas stations', 'tunnel', 'undefined']
TimeofDays = ['dawn/dusk', 'daytime', 'night', 'undefined']


save_dir = '/rscratch/data/bdd100k/domain_clusters_val_time'

train_rep_dir = "/rscratch/data/bdd100k/domain_embedding_train_single"
val_rep_dir = "/rscratch/data/bdd100k/domain_embedding_val_single"

def get_domain_idx(attributes:dict) -> int:

    weather, scene, timeofday = attributes['weather'], attributes['scene'], attributes['timeofday']

    w, s, t = Weathers.index(weather), Scenes.index(scene), TimeofDays.index(timeofday)

    #domain_idx = w * len(Scenes) * len(TimeofDays) + s * len(TimeofDays) + t
    domain_idx = t

    return domain_idx

def read_bdd_labels(path):
    with open(path, 'r') as f:
        bdd = json.load(f)
    return bdd

def get_center(x):

    kmeans_func = KMeans(n_clusters=1, random_state=0)

    kmeans = kmeans_func.fit(x)

    centers = kmeans.cluster_centers_

    center = centers[0]

    return center


if __name__ == "__main__":

    ann_path = bdd_val_label_path
    all_rep_dir = val_rep_dir
    Anns = read_bdd_labels(ann_path)

    name2domain = {}
    with open(os.path.join(all_rep_dir, 'name2domain.json'), 'r') as f:
        name2rep_idx = json.load(f)

    Arrs = {k:[] for k in range(len(TimeofDays))}#

    for ann in Anns:
        img_name = ann['name']
        domain_idx = get_domain_idx(ann['attributes'])
        rep_name = "{}.txt".format(name2rep_idx[img_name])
        rep = np.loadtxt(os.path.join(all_rep_dir, rep_name))

        name2domain[img_name] = domain_idx
        Arrs[domain_idx].append(rep)


    centers = []
    for k, v in Arrs.items():
        if len(v) < 1:
            continue

        center = get_center(v)
        centers.append(center)

    for i, center in enumerate(centers):
        save_name = "{}.txt".format(i)
        print(center)
        np.savetxt(os.path.join(save_dir, save_name), center)

    name2domain_str = json.dumps(name2domain)
    with open(os.path.join(save_dir, 'name2domain.json'), 'w') as f:
        f.write(name2domain_str)