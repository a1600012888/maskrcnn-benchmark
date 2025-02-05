# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "bdd100k_cocofmt_train_time0": {
            "img_dir": "bdd100k/data_part/time-part/images/train/0",
            "ann_file": "bdd100k/data_part/time-part/annotations/train/0.json"
        },
        "bdd100k_cocofmt_val_time0": {
            "img_dir": "bdd100k/data_part/time-part/images/val/0",
            "ann_file": "bdd100k/data_part/time-part/annotations/val/0.json"
        },
        "bdd100k_cocofmt_train_time1": {
            "img_dir": "bdd100k/data_part/time-part/images/train/1",
            "ann_file": "bdd100k/data_part/time-part/annotations/train/1.json"
        },
        "bdd100k_cocofmt_val_time1": {
            "img_dir": "bdd100k/data_part/time-part/images/val/1",
            "ann_file": "bdd100k/data_part/time-part/annotations/val/1.json"
        },
        "bdd100k_cocofmt_train_time2": {
            "img_dir": "bdd100k/data_part/time-part/images/train/2",
            "ann_file": "bdd100k/data_part/time-part/annotations/train/2.json"
        },
        "bdd100k_cocofmt_val_time2": {
            "img_dir": "bdd100k/data_part/time-part/images/val/2",
            "ann_file": "bdd100k/data_part/time-part/annotations/val/2.json"
        },
        "bdd100k_cocofmt_train_time3": {
            "img_dir": "bdd100k/data_part/time-part/images/train/3",
            "ann_file": "bdd100k/data_part/time-part/annotations/train/3.json"
        },
        "bdd100k_cocofmt_val_time3": {
            "img_dir": "bdd100k/data_part/time-part/images/val/3",
            "ann_file": "bdd100k/data_part/time-part/annotations/val/3.json"
        },

        "embed_jitter_domain_bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            'embedding_dir': 'bdd100k/domain_embedding',
            'jitter_range': 0.3,
            'raw_embedding_dir': 'bdd100k/raw_domain_embedding',
        },
        "domain_cluster_bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            'embedding_dir': 'bdd100k/domain_cluster',
        },

        "domain_cluster_bdd100k_cocofmt_val": {
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
            'embedding_dir': 'bdd100k/domain_cluster_val',
        },
        "domain_single_bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            'embedding_dir': 'bdd100k/domain_embedding_single',
        },
        "domain_single_bdd100k_cocofmt_val": {
            # "img_dir": "bdd100k/train",
            # "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            # 'embedding_dir': 'bdd100k/domain_embedding',
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
            'embedding_dir': 'bdd100k/domain_embedding_val_single',
        },
        "domain_idx_bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            'embedding_dir': 'bdd100k/domain_idx_embedding',
        },
        "domain_idx_bdd100k_cocofmt_val": {
            # "img_dir": "bdd100k/train",
            # "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            # 'embedding_dir': 'bdd100k/domain_embedding',
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
            'embedding_dir': 'bdd100k/domain_idx_embedding_val',
        },
        "domain_random_bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            'embedding_dir': 'bdd100k/domain_embedding_random',
        },
        "domain_random_bdd100k_cocofmt_val": {
            #"img_dir": "bdd100k/train",
            #"ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
            'embedding_dir': 'bdd100k/domain_embedding_val_random',
        },
        "domain_bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            'embedding_dir': 'bdd100k/domain_embedding',
        },
        "domain_bdd100k_cocofmt_val": {
            #"img_dir": "bdd100k/train",
            #"ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            #'embedding_dir': 'bdd100k/domain_embedding',
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
            'embedding_dir': 'bdd100k/domain_embedding_val',
        },
        "domain_bdd100k_cocofmt_val_new": {
            # "img_dir": "bdd100k/train",
            # "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
            # 'embedding_dir': 'bdd100k/domain_embedding',
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
            'embedding_dir': 'bdd100k/domain_embedding_val_new',
        },
        "bdd100k_cocofmt_train": {
            "img_dir": "bdd100k/train",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json"
        },
        "bdd100k_cocofmt_val": {
            "img_dir": "bdd100k/val",
            "ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json"
            #"img_dir": "bdd100k/train",
            #"ann_file": "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        }
    }

    @staticmethod
    def get(name, self_str = ''):

        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            if self_str is not '':
                _, dir_name, number = self_str.split("_")
                is_val = name.split('_')[-1]
                img_dir = os.path.join(data_dir,
                                       'bdd100k/data_part',
                                       dir_name, "images",
                                       is_val, '{}'.format(number))
                ann_file = os.path.join(data_dir,
                                       'bdd100k/data_part',
                                       dir_name, "annotations",
                                       is_val,
                                        '{}.json'.format(number))
                args = dict(
                    root=img_dir,
                    ann_file=ann_file,
                )

            else:
                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    root=os.path.join(data_dir, attrs["img_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_file"]),
                )
            if 'domain' in name:
                args['embedding_dir'] = os.path.join(data_dir, attrs['embedding_dir'])

                if 'embed_jitter' in name:
                    args['raw_embedding_dir'] = os.path.join(data_dir, attrs['raw_embedding_dir'])
                    args['jitter_range'] = attrs['jitter_range']
                    return dict(
                        factory="EmbedJitterDomainDataset",
                        args=args,
                    )
                if 'idx' in name:
                    return dict(
                        factory='DomainIdxDataset',
                        args=args
                    )

                return dict(
                    factory="DomainDataset",
                    args=args,
                )
            else:
                return dict(
                    factory="COCODataset",
                    args=args,
                )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
