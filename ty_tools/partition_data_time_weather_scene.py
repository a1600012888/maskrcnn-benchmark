import os
import json

source_dir = '/data/zhangtianyuan/bdd100k/images/100k/train'
target_dir = '/data/zhangtianyuan/bdd100k/images/100k/time-weather-scene-part/train'


Weathers = ['clear', 'partly cloudy', 'overcast', 'rainy', 'snowy', 'foggy', 'undefined']
Scenes = ['residential', 'highway', 'city street', 'parking lot', 'gas stations', 'tunnel', 'undefined']
TimeofDays = ['dawn/dusk', 'daytime', 'night', 'undefined']


json_path = "/data/zhangtianyuan/bdd100k/labels/bdd100k_labels_images_train.json"
coco_json_path = "/data/zhangtianyuan/bdd100k/labels_cocoformat/bdd100k_labels_images_det_coco_train.json"

target_coco_json_dir = "/data/zhangtianyuan/bdd100k/labels_cocoformat/time-weather-scene-part/train"

def get_domain_idx(attributes):
    weather, scene, timeofday = attributes['weather'], attributes['scene'], attributes['timeofday']

    w, s, t = Weathers.index(weather), Scenes.index(scene), TimeofDays.index(timeofday)

    #idx = t + w * len(timeofday)
    #idx = t + w * len(TimeofDays)
    idx = w * len(Scenes) * len(TimeofDays) + s * len(TimeofDays) + t
    print(idx)
    return idx

if __name__ == "__main__":


    domain_idx_list = []
    image_id_2_image_id = {}
    image_id_2_domain_id = {}

    with open(json_path, 'r') as f:
        dics = json.load(f)

    for dic in dics:
        name = dic['name']
        domain_idx = get_domain_idx(dic['attributes'])
        domain_idx_list.append(domain_idx)

        source_name = os.path.join(source_dir, name)
        target_name = os.path.join(target_dir, str(domain_idx), name)
        target_domain_dir = os.path.join(target_dir, str(domain_idx))
        if not os.path.exists(target_domain_dir):
            os.mkdir(target_domain_dir)

        os.link(source_name, target_name)


    with open(coco_json_path, 'r') as f:
        coco_dic = json.load(f)


    new_coco_dics = []
    for i in range(196+1):
        new_coco_dics.append({"categories":coco_dic["categories"],
                              "type":coco_dic["type"],
                              "images":[],
                              'annotations':[]})

    for i, img in enumerate(coco_dic['images']):
        domain_idx = domain_idx_list[i]
        image_id = len(new_coco_dics[domain_idx]['images']) + 1

        image_id_2_image_id[img["id"]] = image_id
        image_id_2_domain_id[img['id']] = domain_idx
        img["id"] = image_id
        new_coco_dics[domain_idx]["images"].append(img)


    for ann in coco_dic["annotations"]:
        image_id = ann["image_id"]
        new_image_id = image_id_2_image_id[image_id]
        domain_idx = image_id_2_domain_id[image_id]
        ann["image_id"] = new_image_id

        new_coco_dics[domain_idx]['annotations'].append(ann)


    for i, new_coco_dic in enumerate(new_coco_dics):
        j = i+1
        target_path = os.path.join(target_coco_json_dir,
                                   "{}.json".format(i))

        with open(target_path, 'w') as f:
            json.dump(new_coco_dic, f)
