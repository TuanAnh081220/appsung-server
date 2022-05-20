import os
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from transformers import TFVisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer

from model.image_captioning.data import get_dataloader


feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    def __init__(self, name='custom_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true != 1), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask))


def download_pretrained():
    model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        'google/vit-base-patch16-224-in21k', 'vinai/phobert-base'
    )
    model.save_pretrained('pretrained.h5')


def get_model():
    # if not os.path.exists('./pretrained.h5'):
    #     download_pretrained()
    # model = TFVisionEncoderDecoderModel.from_pretrained('./pretrained.h5')
    # input_ids_text = tf.keras.layers.Input(shape=(None,), name='input_ids_text', dtype='int32')
    # input_image = tf.keras.layers.Input(shape=(3, 224, 224), name='input_image', dtype='float64')
    # inner = model(input_image, input_ids_text)[0]
    # output = tf.keras.layers.Softmax()(inner)
    # model = tf.keras.Model([input_ids_text, input_image], output)
    model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        'google/vit-base-patch16-224-in21k', 'vinai/phobert-base'
    )
    return model


def train():
    def scheduler(epoch, lr):
        if epoch < 1:
            return 3e-5
        elif (epoch < 3) and (epoch >= 1):
            return 2e-5
        elif (epoch < 6) and (epoch >= 3):
            return 1e-5
        elif (epoch < 10) and (epoch >= 6):
            return 1e-6
        else:
            return 5e-7
    train_loader = get_dataloader()
    if not os.path.exists('./model_checkpoints'):
        try:
            og_umask = os.umask(0)
            os.makedirs('./model_checkpoints', 0o777)
        finally:
            os.umask(og_umask)
    model = get_model()
    loss = CustomNonPaddingTokenLoss()
    adam = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=adam, loss=loss
    )
    file_path = './model_checkpoints/epoch_{epoch:01d}.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                                       save_weights_only=True)
    model.fit(train_loader,
              verbose=1,
              epochs=2,
              callbacks=[model_checkpoint_callback, tf.keras.callbacks.LearningRateScheduler(scheduler)],
              initial_epoch=0)
    model.save('local_trained_model.h5')


def captions_predict(path_image, model, _feature_extractor, _tokenizer):
    try:
        img = tf.keras.preprocessing.image.load_img(path_image)
        img_feature = _feature_extractor(img,return_tensors='np')['pixel_values']
    except:
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_feature = _feature_extractor(img,return_tensors='np')['pixel_values']
    result = model.generate(img_feature, max_length=30, num_beams=5, bos_token_id=0,eos_token_id=2,pad_token_id=1).numpy().tolist()[0]
    result = _tokenizer.decode(result,skip_special_tokens =True)
    return result.replace('@@ ','').replace('_',' ')


def get_pretrained_model():
    # if not os.path.exists('local_trained_model.h5'):
    #     train()
    model = get_model()
    # model = model.from_pretrained('local_trained_model.h5')
    return model