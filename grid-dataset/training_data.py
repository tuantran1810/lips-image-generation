import os
import cv2
import pickle
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import librosa

class TrainingDataConverter:
    def __init__(
            self,
            videoRootFolder,
            audioRootFolder,
            videoExt,
            audioExt,
            outputfolder,
        ):

        self.__paths = dict()

        totalvideo = 0
        identity_set = set()
        for path, _ , files in os.walk(videoRootFolder):
            for name in files:
                code, ext = name.split('.')
                if ext == videoExt:
                    identity = path.split('/')[-1]
                    if identity not in self.__paths:
                        self.__paths[identity] = dict()
                    idmap = self.__paths[identity]
                    idmap[code] = dict()
                    idmap[code]['video'] = os.path.join(path, name)
                    totalvideo += 1
                    identity_set.add(identity)

        for identity in identity_set:
            audio_identity = os.path.join(outputfolder, identity)
            Path(audio_identity).mkdir(parents=True, exist_ok=True)

        totalaudio = 0
        for path, _ , files in os.walk(audioRootFolder):
            for name in files:
                code, ext = name.split('.')
                if ext == audioExt:
                    path_parts = path.split('/')
                    identity = path_parts[-1]
                    training_path = os.path.join(outputfolder, identity)
                    if identity not in self.__paths:
                        print(f"identity {identity} not found, ignore")
                        continue
                    idmap = self.__paths[identity]
                    if code not in idmap:
                        continue
                    idmap[code]['audio'] = os.path.join(path, name)
                    idmap[code]['training'] = os.path.join(training_path, code + '.' + 'pkl')
                    totalaudio += 1
        
        # print(self.__paths)
        print(f"total audio = {totalaudio}, total video = {totalvideo}")

    def __iterate_frames(self, videofile):
        vidcap = cv2.VideoCapture(videofile)
        while True:
            success, image = vidcap.read()
            if not success:
                return
            if image is None:
                print("image is None")
            yield image

    def __audio_logmel_segments(self, wave, n_fft = 1024, n_mels = 128, fsampling = 44100, nsegments = 75):
        wave = wave.astype(float)
        wave_length = wave.shape[0]
        chunk_length = int(wave_length/nsegments)
        for i in range(0, nsegments):
            chunk = wave[(i*chunk_length):((i+1)*chunk_length)]
            s = librosa.feature.melspectrogram(chunk, sr = fsampling, n_fft = n_fft, n_mels=n_mels, fmax= 16000)
            yield librosa.amplitude_to_db(s)

    def run(self):
        for _, content in self.__paths.items():
            for _, info in content.items():
                video_path = info['video']
                audio_path = info['audio']
                training_path = info['training']

                frames = np.stack(
                    [f for f in map(
                        lambda x: np.transpose(x, (2, 1, 0)), 
                        self.__iterate_frames(video_path),
                    )], 
                    axis = 0,
                )

                _, wave = wavfile.read(audio_path)
                audio_logmel = np.stack(
                    [f for f in self.__audio_logmel_segments(wave)],
                    axis = 0,
                )

                data = dict()
                data['video'] = frames
                data['audio'] = audio_logmel

                with open(training_path, 'wb') as fd:
                    pickle.dump(data, fd)
                print(training_path)

def main():
    conv = TrainingDataConverter(
        "/media/tuantran/UbuntuData/dataset/GRID/mouth_videos",
        "/media/tuantran/UbuntuData/dataset/GRID/audio",
        "mp4",
        "wav",
        "/media/tuantran/UbuntuData/dataset/GRID/mouth_images",
    )

    conv.run()

if __name__ == "__main__":
    main()
