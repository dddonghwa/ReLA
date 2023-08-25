import time
import copy
import logging

import matplotlib.pyplot as plt    
import numpy as np
import torch
import os
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.visualizer import Visualizer

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

__all__ = ["RefCOCOMosaicMapper", "MosaicVisualization"]


import math
from PIL import Image, ImageDraw, ImageFilter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .dataset_mappers.refcoco_mapper import RefCOCOMapper, build_transform_train, build_transform_test
from .datasets.refer import REFER
from .datasets.grefer import G_REFER 



class aug:
    img_size = 480
    num_bgs = 4
    aug_prob = 0.5
    tgt_selection = 'fixed' # which target to use among the four.
    blur = False # Blur the rest cells
    move_crs_pnt = False  # mosaic cross point moves freely if True
    

# class mos_args:
    # split = 'test'
    # refer_data_root = '/home/donghwa/data/projects/donghwa/RIS/ReLA/datasets/'
    # dataset = 'refcocog' # refcoco refcocog
    # split_by = 'umd' # unc umd good
    # img_size = 480
    # aug = aug() 


def MosaicVisualization(dataloader, dirname="coco-aug-data-vis", n_sample=2):

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    denorm = A.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        max_pixel_value=1.0
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sent_idx = 0
    os.makedirs(dirname, exist_ok=True)
    # dataloader = build_detection_train_loader(cfg, mapper=mapper)
    it = iter(dataloader)
    batch = next(it)
    n_sample = random.randint(1, len(batch))
    if 'gt_masks' in batch[0].keys():
        for i in range(n_sample):
            data = batch[i]
            img = data['image'].unsqueeze(0)
            img_np = np.transpose(img[0].cpu().numpy(), (1,2,0))
            img_denorm = denorm(image=img_np)['image']
            img_ndarray = (img_denorm*255).astype(np.uint8)
            seg_target = data['gt_masks'].squeeze(0)
            tensor_embedding = data['lang_tokens'][:,:]
            sentences = tokenizer.decode(tensor_embedding[0], skip_special_tokens=True)
            # tokens = [ds.tokenizer.decode([w], skip_special_tokens=False) for w in tensor_embedding[0]]
            # tokens = [x for x in tokens if x!='[PAD]']
            
            fpath = os.path.join(dirname, os.path.basename(data["file_name"]))
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(img_ndarray)
            ax1.set_xlabel("Mosaic Image")
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(seg_target)
            ax2.set_xlabel("Segmentation Map")
            plt.suptitle(sentences)
            plt.tight_layout()
            plt.savefig(fpath)
            
    else :
        
        for i in range(n_sample):
            d = batch[i]
            img = np.array(Image.open(d["file_name"]))
            visualizer = Visualizer(img, metadata={})
            vis = visualizer.draw_dataset_dict(d)
            fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
            vis.save(fpath)



    
class RefCOCOMosaicMapper():
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        dataset,
        split_by,
        split,
        refer_data_root,
        aug,
        bert_type,
        max_tokens,
        merge=True,
    ):
        
        # self.is_train = is_train
        self.merge = merge
        # self.tfm_gens = tfm_gens
        # logging.getLogger(__name__).info(
        #     "Full TransformGens used: {}".format(str(self.tfm_gens))
        # )
        self.dataset = dataset
        self.split_by = split_by
        self.split = split
        self.refer_data_root = refer_data_root
        
        self.bert_type = bert_type
        self.max_tokens = max_tokens
        logging.getLogger(__name__).info(
            "Loading BERT tokenizer: {}...".format(self.bert_type)
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)

        # self.img_format = image_format
        
        
        self.classes = []
        self.aug = aug   
        self.bert_type = bert_type   
        
        self.img_sz = self.aug.img_size
         
        each_img_sz = int(self.aug.img_size/math.sqrt(self.aug.num_bgs))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        self.resize_bg1 = A.Compose([
            A.Resize(self.aug.img_size, self.aug.img_size, always_apply=True)])
        
        self.resize_bg4 = A.Compose([
            A.Resize(each_img_sz, each_img_sz, always_apply=True)],
            additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image',
                                'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask',})
            
        self.transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2 (),
        ])
        
        
        # load annotations
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.refer = REFER(self.refer_data_root, self.dataset, self.split_by)
        elif dataset in ['grefcoco']:
            self.refer = G_REFER(self.refer_data_root, self.dataset, self.split_by) 
        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids#[:500]
        self.ref_id2idx = dict(zip(ref_ids, range(len(ref_ids))))
        self.ref_idx2id = dict(zip(range(len(ref_ids)), ref_ids))
        self.img2refs = self.refer.imgToRefs


        # self.tokenizer.add_special_tokens({'additional_special_tokens': task_tokens})
        # self.tokenizer.add_tokens(position_tokens)

        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        self.max_tokens = 20
        self.is_train = is_train
        self.input_ids = []
        self.attention_masks = []
        for i, r in enumerate(ref_ids):
            ref = self.refer.Refs[r]
            
            sentences_for_ref = []
            attentions_for_ref = []
            for j, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True, max_length=self.max_tokens, truncation=True)
                #input_ids = input_ids[:self.max_tokens]
                padded_input_ids = [0] * self.max_tokens
                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask = [0] * self.max_tokens 
                attention_mask[:len(input_ids)] = [1]*len(input_ids)
                sentences_for_ref.append(padded_input_ids)
                attentions_for_ref.append(attention_mask)

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            
        if self.aug.blur:
            self.blur = ImageFilter.GaussianBlur(100)
        np.random.seed()
        
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            tfm_gens = build_transform_train(cfg)
        else:
            tfm_gens = build_transform_test(cfg)

        assert len(cfg.DATASETS.TEST) == 1
        assert type(cfg.DATASETS.TEST) == tuple
        dataset, split_by, split = cfg.DATASETS.TEST[0].split('_')
        ret = {
            "is_train": is_train,
            "dataset" : dataset,
            "split_by" : split_by,
            "split" : 'train' if is_train else split,
            "refer_data_root" : cfg.DATASETS.REF_ROOT,
            "aug" : aug(),
            "bert_type": cfg.REFERRING.BERT_TYPE,
            "max_tokens": cfg.REFERRING.MAX_TOKENS,
        }
        return ret
    

    @staticmethod
    def _merge_masks(x):
        return x.sum(dim=0, keepdim=True).clamp(max=1)
   
        
    def __call__(self, dataset_dict):
        
        dataset_dict = copy.deepcopy(dataset_dict)
        img_id = dataset_dict["image_id"]
        index = dataset_dict["id"]
        
        # decide mosaic size
        if self.split=='train':
            if self.aug.num_bgs==4:
                aug_prob = self.aug.aug_prob
                num_bgs = np.random.choice([1, 4], p=[1-aug_prob, aug_prob])
            else:
                num_bgs = 1
        else:
            num_bgs = 1
        
        
        target_sent_idx = np.random.choice(len(self.input_ids[index])) 
        
        if num_bgs==1:
            ref_ids = []
            sent_idxs = []
            sents = np.array([], dtype='str')
            
        else:                  
            ref_ids = list(np.random.choice(self.ref_ids, size=num_bgs-1, replace=False))
            sent_idxs = [np.random.choice(len(self.refer.Refs[r]['sentences'])) for r in ref_ids]
            sents = np.array([self.refer.Refs[r]['sentences'][sent_idxs[i]]['raw'] for i, r in enumerate(ref_ids)], dtype='str')
        
        insert_idx = np.random.choice(range(num_bgs))
        ref_ids = np.insert(ref_ids, insert_idx, self.ref_idx2id[index]).astype(int)
        sents = np.insert(sents, insert_idx,
                          self.refer.Refs[ref_ids[insert_idx]]['sentences'][target_sent_idx]['raw'])
        sent_idxs = np.insert(sent_idxs, insert_idx, target_sent_idx).astype(int)
        
        # pick a target origin
        if self.aug.tgt_selection == 'random':
            target_idx = np.random.choice(range(num_bgs))
            target_ref_idx = self.ref_id2idx[ref_ids[target_idx]]
            target_sent_idx = int(np.random.choice(len(self.input_ids[target_ref_idx])))
        elif self.aug.tgt_selection == 'longest':
            target_idx = np.argmax(list(map(len, sents)))
            target_sent_idx = sent_idxs[target_idx]
        elif self.aug.tgt_selection == 'fixed':
            target_idx = insert_idx
        target_ref_id = ref_ids[target_idx]

        # load items
        imgs, masks = [], []
        for ref_id in ref_ids:
            img_id = self.refer.getImgIds([ref_id])[0]
            img_info = self.refer.Imgs[img_id]
            img_path = os.path.join(self.refer.IMAGE_DIR, img_info['file_name'])
            img = Image.open(img_path).convert("RGB")
            imgs.append(np.array(img))
            ref = self.refer.loadRefs(ref_ids=[ref_id])
            if self.dataset in ['refcoco', 'refcoco+', 'refcocog']:
                mask = np.array(self.refer.getMask(ref[0])['mask'])
            elif self.dataset in ['grefcoco'] : 
                mask = self.refer.getMaskByRef(ref[0], ref_id, self.merge)['mask']
            masks.append(mask)
        
        # image resize and apply 4in1 augmentation
        if num_bgs==1:
            resized = self.resize_bg1(image=imgs[0], mask=masks[0])
            imgs, masks = [resized['image']], [resized['mask']]
            img = imgs[0]
        else:
            if self.aug.move_crs_pnt:
                crs_y = np.random.randint(0, self.img_sz+1)
                crs_x = np.random.randint(0, self.img_sz+1)
            else:
                crs_y = 480//2 #
                crs_x = 480//2 #

            if crs_y==0 or crs_x==0:
                img1 = np.zeros([0,crs_x,3]) if crs_y==0 else np.zeros([crs_y,0,3])
                mask1 = np.zeros([0,crs_x]) if crs_y==0 else np.zeros([crs_y,0])
            else:
                resize_bg1 = A.Compose([A.Resize(crs_y, crs_x, always_apply=True)])
                temp = resize_bg1(image=imgs[0], mask=masks[0])
                img1 = temp['image']
                mask1 = temp['mask']
            
            if crs_y==0 or crs_x==self.img_sz:
                img2 = np.zeros([0,self.img_sz-crs_x,3]) if crs_y==0 \
                    else np.zeros([crs_y,0,3])
                mask2 = np.zeros([0,self.img_sz-crs_x]) if crs_y==0 \
                    else np.zeros([crs_y,0])
            else:
                resize_bg2 = A.Compose([
                    A.Resize(crs_y, self.img_sz-crs_x, always_apply=True)])
                temp = resize_bg2(image=imgs[1], mask=masks[1])
                img2 = temp['image']
                mask2 = temp['mask']
                
            if crs_y==self.img_sz or crs_x==0:
                img3 = np.zeros([0,crs_x,3]) if crs_y==self.img_sz \
                    else np.zeros([self.img_sz-crs_y,0,3])
                mask3 = np.zeros([0,crs_x]) if crs_y==self.img_sz \
                    else np.zeros([self.img_sz-crs_y,0])
            else:                
                resize_bg3 = A.Compose([
                    A.Resize(self.img_sz-crs_y, crs_x, always_apply=True)])
                temp = resize_bg3(image=imgs[2], mask=masks[2])
                img3 = temp['image']
                mask3 = temp['mask']
                
            if crs_y==self.img_sz or crs_x==self.img_sz:
                img4 = np.zeros([0,self.img_sz-crs_x,3]) if crs_y==self.img_sz \
                    else np.zeros([self.img_sz-crs_y,0,3])
                mask4 = np.zeros([0,self.img_sz-crs_x]) if crs_y==self.img_sz \
                    else np.zeros([self.img_sz-crs_y,0])
            else:
                resize_bg4 = A.Compose([
                    A.Resize(self.img_sz-crs_y,
                             self.img_sz-crs_x, always_apply=True)])
                temp = resize_bg4(image=imgs[3], mask=masks[3])
                img4 = temp['image']
                mask4 = temp['mask']
                
            imgs = [img1, img2, img3, img4]
            masks = [mask1, mask2, mask3, mask4]
            
            # scale effect ablation
            if self.aug.blur:
                imgs = [np.asarray(Image.fromarray(x).filter(self.blur)) if i!=insert_idx else x for i, x in enumerate(imgs)]
            
            num_rows = num_cols = int(math.sqrt(num_bgs))
            idxs = [(i*num_cols,i*num_cols+num_cols) for i in range(num_rows)]
            img = [np.concatenate(imgs[_from:_to], axis=1) for (_from, _to) in idxs]
            img = np.concatenate(img, axis=0).astype(np.uint8)
            
            masks_arr = []
            for bg_idx in range(num_bgs):
                mask = masks[bg_idx]
                temp = [mask if idx==bg_idx else np.zeros_like(masks[idx]) for idx in range(num_bgs)]
                mask = [np.concatenate(temp[_from:_to], axis=1) for (_from, _to) in idxs]
                mask = np.concatenate(mask, axis=0).astype(np.int32)
                masks_arr.append(mask)
            masks = masks_arr
        
        mask = masks[target_idx]    
        mask = mask.astype(np.uint8)
        mask[mask>0] = 1

        item = self.transforms(image=img, mask=mask)
        img_tensor = item['image']
        target = item['mask'].long()

        target_ref_idx = self.ref_id2idx[target_ref_id]
        # if self.is_train:
        #     embedding = []
        #     att = []
        #     for s in range(len(self.input_ids[target_ref_idx])):
        #         padded_input_ids = self.input_ids[target_ref_idx][s]
        #         tensor_embeddings = torch.tensor(padded_input_ids).unsqueeze(0)
        #         attention_mask = self.attention_masks[target_ref_idx][s]
        #         attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        #         embedding.append(tensor_embeddings.unsqueeze(-1))
        #         att.append(attention_mask.unsqueeze(-1))
        #     tensor_embeddings = torch.cat(embedding, dim=-1)
        #     attention_mask = torch.cat(att, dim=-1)
        # else:
        padded_input_ids = self.input_ids[target_ref_idx][target_sent_idx]
        tensor_embeddings = torch.tensor(padded_input_ids).unsqueeze(0)
        attention_mask = self.attention_masks[target_ref_idx][target_sent_idx]
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        empty = dataset_dict.get("empty", False)
        dataset_dict["empty"] = empty
        dataset_dict['image'] = img_tensor
        dataset_dict['gt_masks'] = target.unsqueeze(0)
        dataset_dict['lang_tokens'] = tensor_embeddings
        dataset_dict['lang_mask'] = attention_mask
        # dataset_dict["gt_mask_merged"] = self._merge_masks(target) if self.merge else None
        dataset_dict["gt_mask_merged"] = target.unsqueeze(0)

        
        # item = {
        #     'image': img_tensor,
        #     'seg_target': target,
        #     'sentence': tensor_embeddings,
        #     'attn_mask': attention_mask,
        # }
        return dataset_dict

    

