import os
import cv2
import json
import math
import tqdm
import gdown
import pickle
import requests
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from vncorenlp import VnCoreNLP
from transformers import AutoFeatureExtractor, AutoTokenizer


def download_data():
    gdown.download('https://drive.google.com/uc?id=1YexKrE6o0UiJhFWpE8M5LKoe6-k3AiM4')
    unzip_cmd = 'unzip ./UIT-ViIC-20200417T021508Z-001.zip'
    print(unzip_cmd)
    os.system(unzip_cmd)
    cmd_list = [
        'mkdir -p vncorenlp/models/wordsegmenter',
        'wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar',
        'wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab',
        'wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr',
        'mv VnCoreNLP-1.1.1.jar vncorenlp/',
        'mv vi-vocab vncorenlp/models/wordsegmenter/',
        'mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/'
    ]
    for cmd in cmd_list:
        print(cmd)
        os.system(cmd)


def get_data():
    if not os.path.exists('./vncorenlp/VnCoreNLP-1.1.1.jar'):
        download_data()
    annotator = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
    with open('./UIT-ViIC/uitviic_captions_train2017.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['annotations'])
    df['id'] = df['image_id']
    df['captions'] = df['caption']
    df = df[['id', 'captions']]
    df = df.join(pd.DataFrame(data['images'])[['id', 'coco_url']].set_index('id'), on='id')
    df = df[~df['coco_url'].isin(['http://images.cocodataset.org/train2017/000000401901.jpg',
                                  'http://images.cocodataset.org/train2017/000000577207.jpg',
                                  'http://images.cocodataset.org/train2017/000000325387.jpg',
                                  'http://images.cocodataset.org/train2017/000000186888.jpg'])]
    df['captions'] = df['captions'].apply(lambda x: ' '.join(annotator.tokenize(x.lower())[0]))
    return df.to_dict('records')


class ImageCaptioiningDataloader(tf.keras.utils.Sequence):
    def __init__(self, data, path_image_train, image_feature_extractor='ViT', language_embedding='PhoBERT',
                 model_flow='encoder_decoder', batch_size=16, max_length=36):
        self.batch_size = batch_size
        self.image = []
        self.feature_image = {}
        self.input_ids = []
        self.input_mask = []
        self.output = []

        self.image_feature_extractor = image_feature_extractor
        self.language_embedding = language_embedding
        self.model_flow = model_flow

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_processor = self.ViT_processor

        if language_embedding == 'FastText':
            # with open(r"/content/fasttext_embedding.pkl", "rb") as f:
            #     embedding_data_1 = pickle.load(f)
            with open(r"/content/drive/MyDrive/Best_model/fasttext_embedding.pkl", "rb") as f:
                embedding_data_1 = pickle.load(f)
            self.embedding_data = {}
            for i in embedding_data_1:
                self.embedding_data[str(i)] = embedding_data_1[i]

            self.tokenizer = tf.keras.layers.TextVectorization(
                max_tokens=len(self.embedding_data.keys()))
            self.tokenizer.adapt(['<start>', '<end>'] + list(self.embedding_data.keys()))
        elif language_embedding == 'PhoBERT':
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        else:
            raise Exception("Wrong language embedding")

        for i in data:
            if image_feature_extractor == 'ViT':
                if not i['id'] in self.feature_image:
                    if len(i.keys()) == 2:
                        self.feature_image[i['id']] = self.image_processor(path_image_train + i['id'], None)
                    else:
                        self.feature_image[i['id']] = self.image_processor(None, i['coco_url'])

            if model_flow == 'encoder_decoder':
                self.input_ids.append(i['captions'])
                self.image.append(i['id'])

            elif model_flow == 'merging' and language_embedding == 'PhoBERT':
                temp = self.tokenizer(i['captions'], truncation=True, return_token_type_ids=False,
                                      return_attention_mask=False)['input_ids']
                for j in range(1, len(temp)):
                    self.input_ids.append(temp[:j] + [1] * (max_length - j))
                    if image_feature_extractor == 'ViT':
                        self.image.append(i['id'])
                    else:
                        self.image.append(path_image_train + i['id'])

                    self.input_mask.append([1] * j + [0] * (max_length - j))
                    self.output.append(temp[j])

            elif model_flow == 'merging' and language_embedding == 'FastText':
                temp = tf.convert_to_tensor(self.tokenizer('start ' + i['captions'] + ' end'))
                for j in range(1, len(temp)):
                    if temp[j] == 0:
                        break
                    self.input_ids.append(tf.concat([temp[:j], [0] * (max_length - j)], 0))
                    self.image.append(i['id'])
                    self.output.append(temp[j])

        if model_flow == 'encoder_decoder':
            self.input_ids = self.tokenizer(self.input_ids, truncation=True, padding=True, return_tensors="np",
                                            return_token_type_ids=False, return_attention_mask=False).input_ids

    def __len__(self):
        return math.ceil(len(self.input_ids) / self.batch_size)

    def ViT_processor(self, path=None, url=None):
        if path:
            try:
                img = tf.keras.preprocessing.image.load_img(path)
                img_feature = self.feature_extractor(img, return_tensors='np')['pixel_values'][0]
            except:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_feature = self.feature_extractor(img, return_tensors='np')['pixel_values'][0]
        else:
            img = Image.open(requests.get(url, stream=True).raw)
            img_feature = self.feature_extractor(img, return_tensors='np')['pixel_values'][0]
        return img_feature

    def __getitem__(self, idx):
        if self.model_flow == 'encoder_decoder':
            batch_input_ids = self.input_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_image_feature = [self.feature_image[i] for i in
                                   self.image[idx * self.batch_size:(idx + 1) * self.batch_size]]
            return [batch_input_ids[:, :-1], np.array(batch_image_feature)], batch_input_ids[:, 1:]
        else:
            batch_input_ids = self.input_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_output = self.output[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_image_feature_extractor = [self.feature_image[i] for i in
                                             self.image[idx * self.batch_size:(idx + 1) * self.batch_size]]
            batch_input_mask = self.input_mask[idx * self.batch_size:(idx + 1) * self.batch_size]
            return [np.array(batch_input_ids), np.array(batch_input_mask),
                    np.array(batch_image_feature_extractor)], np.array(batch_output)


def get_dataloader():
    train_data = get_data()[:500]
    image_path = '/images/'
    dataloader = ImageCaptioiningDataloader(train_data, image_path)
    return dataloader
