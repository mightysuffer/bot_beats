import numpy as np
import librosa
import os
from pytube import YouTube
import subprocess
import telebot

token = ''
bot = telebot.TeleBot(token)  # Token
folder = ''  # Path to save temp files
os.chdir(folder)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Отправь ссылку на YT бит")
    else:
        try:
            beat_path = download_yt(link=message.text)
            back = song_info(beat_path)
            doc = open(folder + beat_path, 'rb')
            bot.send_audio(message.from_user.id, doc, back)
            doc.close()
            os.remove(beat_path)
        except Exception as ex:
            bot.send_message(message.from_user.id, "Ошибка, попробуй заново")


def download_yt(link):
    yt = YouTube(link)
    video = yt.streams.filter(only_audio=True).desc().first()
    video.download(folder)
    filename, ext = os.path.splitext(video.default_filename)
    input_path = f"\"{folder}{filename}{ext}\""
    output_path = f"\"{folder}{filename}.wav\""
    ffmpeg_path = ""
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
        answer = "Likely key: " + max(self.key_dict, key=self.key_dict.get) + ", correlation: " + str(self.bestcorr)
        if self.altkey is not None:
            answer += "  also possible: " + self.altkey + ", correlation: " + str(self.altbestcorr)
        return answer


def song_info(beat_path):
    y, sr = librosa.load(beat_path, offset=30)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    song = TonalFragment(y_harmonic, sr, 35, 60)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr, start_bpm=110, tightness=100, trim=True)
    key = song.get_key()
    info = "BPM: " + str(tempo) + " \n" + key
    return info


bot.polling(none_stop=True, interval=0)
