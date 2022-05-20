import google.cloud.texttospeech as tts
from google.oauth2 import service_account
import json

from model.image_captioning.model import get_pretrained_model, captions_predict, feature_extractor, tokenizer


def get_caption(file):
    # do something here with model
    model = get_pretrained_model()
    caption = captions_predict(file, model, feature_extractor, tokenizer)
    return caption


def get_caption_filepath(caption):
    # text to speech model
    # voice_name = 'en-US-Wavenet-A'
    f = open('key.json')
    data = json.load(f)
    credentials = service_account.Credentials.from_service_account_info(data)

    voice_name = 'vi-VN-Wavenet-A'
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=caption)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient(credentials=credentials)
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )

    filename = "advertising.mp3"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')
    return filename
