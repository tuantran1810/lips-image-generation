import os
from pathlib import Path

class AudioExtractor:
    def __init__(
            self,
            videoRootFolder,
            audioRootFolder,
            videoExt,
            audioExt,
        ):

        self.__paths = dict()

        totalvideo = 0
        identity_set = set()
        for path, _ , files in os.walk(videoRootFolder):
            for name in files:
                code, ext = name.split('.')
                if ext == videoExt:
                    path_parts = path.split('/')
                    identity = path_parts[-1]
                    audio_path = os.path.join(audioRootFolder, identity)
                    if identity not in self.__paths:
                        self.__paths[identity] = dict()
                    idmap = self.__paths[identity]
                    idmap[code] = dict()
                    idmap[code]['video'] = os.path.join(path, name)
                    idmap[code]['audio'] = os.path.join(audio_path, code + '.' + audioExt)
                    totalvideo += 1
                    identity_set.add(identity)

        for identity in identity_set:
            audio_identity = os.path.join(audioRootFolder, identity)
            Path(audio_identity).mkdir(parents=True, exist_ok=True)

        print(f"total video = {totalvideo}")

    def run(self):
        for _, content in self.__paths.items():
            for _, info in content.items():
                video_path = info['video']
                audio_path = info['audio']
                command = 'ffmpeg -i ' + video_path + ' -ac 1 ' + audio_path
                try:
                    os.system(command)
                except:
                    print(f"exception occurs: {command}")


def main():
    ae = AudioExtractor(
        "/media/tuantran/UbuntuData/dataset/GRID/video",
        "/media/tuantran/UbuntuData/dataset/GRID/audio",
        "mpg",
        "wav",
    )

    ae.run()

if __name__ == "__main__":
    main()
