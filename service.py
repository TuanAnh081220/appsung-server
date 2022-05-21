import google.cloud.texttospeech as tts
from google.oauth2 import service_account
import json
import random

# from model.image_captioning.model import captions_predict

list_caption = ['người đàn ông đang đánh tennis ngoài sân.',
 'một vận động viên tennis đang vung vợt đánh bóng.',
 'một cầu thủ tennis đang vung vợt tennis đỡ bóng.',
 'người đàn ông đang đứng ngoài biên cầm vợt sẵn sàng đỡ bóng.',
 'vận động viên tennis nam đang trong tư thế chuẩn bị đỡ bóng.',
 'một màn hình máy tính trắng đang đặt trên chiếc bàn gỗ.',
 'một cái bàn bừa bộn và có một quả bóng nảy màu vàng dưới đất.',
 'hình ảnh một quả bóng bự bên cạnh bàn gỗ.',
 'một quả bóng nảy vàng đang đặt dưới đất cạnh chiếc bàn gỗ.',
 'một quả bóng yoga vàng đang đặt dưới đất cạnh chiếc bàn.',
 'một người đàn ông đang chơi ném dĩa ngoài sân.',
 'một số người đàn ông đang chơi ném dĩa ngoài sân.',
 'những người đàn ông đang luyện tập ném dĩa.',
 'một số người đàn ông đang cùng nhau chơi ném dĩa.',
 'một số người đàn ông đang luyện tập ném dĩa cùng nhau.',
 'người phụ nữ đang cầm vợt tennis trên sân.',
 'vận động viên tennis nữ đang vung vợt đỡ bóng tennis.',
 'người phụ nữ đang vung vợt tennis chuẩn bị đánh bóng.',
 'người phụ nữ đang cầm vợt tennis thi đấu trên sân.',
 'người phụ nữ đang vung vợt về phía quả bóng tennis.',
 'người phụ nữ áo trắng đang vung vợt đánh bóng tennis.',
 'vận động viên tennis nữ đang cầm vợt đánh bóng trên sân.',
 'vận động viên tennis nữ đang vung vợt đỡ bóng.',
 'người phụ nữ đang đánh tennis trước nhiều người đang quan sát.',
 'người phụ nữ đang thi đấu tennis trước đông đảo khán giả.',
 'hình ảnh trắng đen một nhóm vận động viên tennis.',
 'một nhóm người đàn ông đang cầm vợt tennis chụp ảnh',
 'một nhóm vận động viên tennis đang chụp ảnh trên sân tennis.',
 'hình ảnh một nhóm vận động viên tennis.',
 'hình ảnh trắng đen một nhóm vận động viên tennis trên sân.',
 'người đàn ông và người phụ nữ đang đứng người vườn.',
 'hình ảnh người phụ nữ và người đàn ông cầm vợt tennis đứng ngoài vườn.',
 'hình ảnh trắng đen hai người cầm vợt tennis đứng ngoài vườn',
 'người phụ nữ và người đàn ông cầm vợt tennis tạo dáng chụp ảnh.',
 'hai người đang cầm vợt tennis đứng ngoài vườn chụp ảnh.',
 'người đàn ông cầm vợt với đánh bóng tennis.',
 'vận động viên tennis nam đang vươn vợt đánh bóng tennis.',
 'vận động viên tennis nam đang chạy tới vươn vợt đỡ bóng.',
 'vận động viên tennis nam đang cầm vợt chạy trên sân.',
 'vận động viên tennis nam đang cố gắng đỡ bóng tennis.',
 'người đàn ông áo trắng đang cầm vợt tennis đứng trên sân.',
 'vận động viên tennis nam đang thi đấu trước đông đảo khán giả đang quan sát.',
 'một người đàn ông đang vung vợt đỡ bóng tennis.',
 'vận động viên tennis nam trong tư thế chuẩn bị đánh bóng.',
 'vận động viên tennis nam đang cầm vợt chuẩn bị đỡ bóng thấp.',
 'người đàn ông và bé gái đang chơi bóng đá trên sân cỏ.',
 'người đàn ông đứng trên sân cỏ cùng quả bóng và bé gái.',
 'một bé gái đang chơi ném dĩa và người đàn ông quan sát.',
 'một bé gái đang chơi ném dĩa và người đàn ông đang chơi bóng.',
 'người đàn ông đang quan sát một bé gái chơi ném dĩa',
 'hình ảnh chiếc dù , con chó và một quả bóng cam.',
 'cây dù đang đè lên cổ một con chó đang chơi bóng.',
 'cây dù đang đè lên một con chó đang quan sát quả bóng.',
 'một con chó đang chơi bóng dưới một cây dù.',
 'một con chó đang chơi với quả bóng và cây dù.',
 'hình ảnh những cậu bé đang chơi bóng đá trên sân.',
 'một nhóm các cậu bé đang tranh giành quả bóng đá bên hàng rào.',
 'những cậu bé đang chơi bóng đá bên hàng rào.',
 'những đứa trẻ mặc đồ đá bóng đang tụ tập trên sân cỏ.',
 'những đứa trẻ đang cùng nhau đá bóng trên sân cỏ.',
 'người phụ nữ áo trắng đang cầm vợt và bóng tennis.',
 'vận động viên tennis nữ đang cầm vợt chuẩn bị giao bóng.',
 'vận động viên tennis nữ đang cầm vợt và bóng trên tay.',
 'một người phụ nữ đang chơi một trận tennis.',
 'một người phụ nữ đang trong tư thế chuẩn bị giao bóng.',
 'hình ảnh một người phụ nữ đang chơi một trận tennis.',
 'vận động viên tennis nữ đang bay đến đón bóng.',
 'người phụ nữ đang chạy trên sân đón bóng tennis.',
 'vận động viên tennis nữ đang thi đấu trên sân và những người đang quan sát.',
 'vận động viên tennis nữ đang chạy ngoài biên chuẩn bị đánh bóng tennis.',
 'một cầu thủ bóng đá đang nhảy lên đánh bóng bằng đầu.',
 'cầu thủ bóng đá nam đang nhảy lên tranh bóng.',
 'cầu thủ bóng đá nam đang tranh bóng trên không.',
 'cầu thủ bóng đá nam đang nhảy lên đón bóng.',
 'cầu thủ bóng đá nam đang bật lên đánh đầu.',
 'hình ảnh ba người đàn ông đang chơi một trận bóng đá.',
 'những người đàn ông đang chơi bóng đá trên sân.',
 'một thủ môn đang phát bóng lên và một cầu thủ đang chạy trên sân.',
 'hình ảnh một thủ môn đang ném bóng về phía trước.',
 'hai người đàn ông đang chơi bóng đá và một trọng tài đang quan sát.',
 'một bé gái đang đeo găng tay bóng chày cầm quả bóng chày.',
 'một bé gái đang dùng găng tay bóng chày che mặt.',
 'người phụ nữ đang che mặt bằng găng tay bóng chày và quả bóng.',
 'người phụ nữ đeo găng tay bóng chày và cầm bóng chuẩn bị phát bóng.',
 'người phụ nữ cầm quả bóng chày chuẩn bị phát bóng.',
 'hình ảnh một trận tennis đang diễn ra.',
 'một trận tennis đang diễn ra với nhiều khán giả đang quan sát.',
 'vận động viên tennis đang nhảy lên đỡ bóng trong một trận tennis.',
 'vận động viên tennis đang giao bóng trong một trận tennis.',
 'hai người đang chơi tennis trên sân trước đông đảo khán giả quan sát.',
 'hai đứa trẻ đang chơi trên chiếc phao.',
 'hai đứa trẻ đang ngồi trên cái phao trong nhà banh.',
 'hai đứa trẻ đang đùa giỡn trên ghế phao trong nhà banh.',
 'hai đứa trẻ đang ngồi trên ghế phao chơi đùa.',
 'hai đứa trẻ đang chơi cùng nhau trên chiếc ghế phao.',
 'con hải cẩu trắng đang chơi với hai quả bóng dưới nước.',
 'một con hải cẩu đang chơi với quả bóng xanh và vàng.',
 'một con hải cầu đang ngậm dây chơi cùng quả bóng.',
 'một con hải cẩu đang bơi dưới nước cùng quả bóng.',
 'hình ảnh con hải cẩu đang chơi với 2 quả bóng phao trên mặt nước.']


def get_caption(file):
    # do something here with model
    # caption = captions_predict(file)
    random_index = random.randint(0, len(list_caption) - 1)
    caption = list_caption[random_index]
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
