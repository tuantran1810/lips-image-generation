import cv2
import json
import moviepy.editor as mpy 
from moviepy.video.fx.all import crop, resize
from pathlib import Path

class VideoCropper(object):
    def __init__(self, input_json, output_path, part, output_size=(96,128)):
        self.__facepos = None
        with open(input_json) as fd:
            self.__facepos = json.load(fd)

        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.__folders = set()
        self.__total_videos = 0
        for file, content in self.__facepos.items():
            folder = file.split('/')[-2]
            self.__folders.add(folder)
            if part in content:
                self.__total_videos += 1

        for fol in self.__folders:
            Path(output_path + fol).mkdir(parents=True, exist_ok=True)

        self.__output_size = output_size
        self.__output_path = output_path
        self.__part = part

    def run(self):
        cnt = 0
        one_percent = self.__total_videos // 100
        for file, content in self.__facepos.items():
            if self.__part in content:
                cnt += 1
                x, y, w, h = content[self.__part]
                tmp = file.split('/')
                filename = tmp[-1]
                code = filename.split('.')[0]
                folder = tmp[-2]

                video = mpy.VideoFileClip(file)
                video = crop(video, x1 = x, y1 = y, x2 = x + w, y2 = y + h)
                video = resize(video, self.__output_size)

                path = "{}{}/{}.mp4".format(self.__output_path, folder, code)
                video.write_videofile(path, audio=False, threads=6, preset='placebo')
            if cnt % one_percent == 0:
                print(f"-------------{cnt/one_percent}% of videos have been processed!---------------")

def main():
    crop_result = "./result.json"
    # VideoCropper(crop_result, "/media/tuantran/UbuntuData/dataset/GRID/mouth_videos/", "mouth", (64, 64)).run()
    VideoCropper(crop_result, "/media/tuantran/UbuntuData/dataset/GRID/face_videos/", "face").run()

if __name__ == "__main__":
    main()
