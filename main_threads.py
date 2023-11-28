import numpy as np
import librosa
import os
from pytube import YouTube
import subprocess
import telebot
import logging
import concurrent.futures
import requests
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO, filename="bot_log.log", filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
token = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(token)  # Token
folder = os.getenv("FOLDER_PATH")
os.chdir(folder)
FFMPEG_PATH = os.getenv("FFMPEG_PATH")
workers = 3
in_work = 0


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if in_work > workers:
        bot.reply_to(message, f"Много битов в обработке, придется подождать, перед вами: {in_work} запросов")
    make_thread(message)


def download_yt(link):
    yt = YouTube(link)
    video = yt.streams.filter(only_audio=True).desc().first()
    video.download(folder)
    filename, ext = os.path.splitext(video.default_filename)
    input_path = f"\"{folder}{filename}{ext}\""
    output_path = f"\"{folder}{filename}.wav\""
    ffmpeg_path = FFMPEG_PATH
    command = f"{ffmpeg_path} -i {input_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {output_path} "
    subprocess.call(command)
    os.remove(video.default_filename)
    return f"{filename}.wav"


class TonalFragment(object):
    def __init__(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend

        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)

        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)}

        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)},
                         **{keys[i + 12]: self.min_key_corrs[i] for i in range(12)}}

        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())

        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr * 0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr

    def get_key(self):
        answer = "Likely key: " + max(self.key_dict, key=self.key_dict.get)
        if self.altkey is not None:
            answer += "  also possible: " + self.altkey
        return answer


def song_info(beat_path):
    y, sr = librosa.load(beat_path, offset=30)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    song = TonalFragment(y_harmonic, sr, 35, 60)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr, start_bpm=110, tightness=100, trim=True)
    key = song.get_key()
    info = "BPM: " + str(tempo) + " \n" + key
    return info


def process_audio(text):
    beat_path = download_yt(text)
    back = song_info(beat_path)
    doc = open(folder + beat_path, 'rb')
    return back, doc, beat_path


def check_link(link):
    try:
        response = requests.get(link)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as ex:
        logging.error(ex, exc_info=True)
        return False


def handle_message(text):
    if text.text == "/start":
        bot.send_message(text.from_user.id, "Отправь ссылку на YT бит")
    else:
        try:
            if check_link(text.text):
                bot.reply_to(text, "Бит в обработке")
                logging.info(f"Request = \"{text.text}\"")
                back, doc, beat_path = process_audio(text.text)
                bot.send_audio(text.from_user.id, doc, back, reply_to_message_id=text.id)
                doc.close()
                os.remove(beat_path)
            else:
                bot.reply_to(text, "Нерабочая ссылка")
        except Exception as ex:
            logging.error(ex, exc_info=True)
            bot.reply_to(text, "Ошибка, попробуй заново")


def make_thread(message):
    global in_work
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            in_work += 1
            executor.submit(handle_message, message)
    except Exception as ex:
        in_work -= 1
        logging.error(ex, exc_info=True)
        bot.send_message(message.from_user.id, "Непредвиденная ошибка, попробуй еще раз")
    else:
        in_work -= 1


bot.polling(none_stop=True, interval=0)
