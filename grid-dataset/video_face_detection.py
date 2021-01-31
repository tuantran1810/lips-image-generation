import cv2
import sys
import traceback
import os
from os import path
import numpy
from facenet_pytorch import MTCNN
import json
import matplotlib.pyplot as plt

class VideoFaceDetector(object):
    def __init__(
            self,
            pre_data,
            rootfolder,
            ext,
            face_expected_ratio,
            mouth_expected_ratio,
            zoomout_ratio = 1.3,
        ):

        self.__paths = dict()
        if pre_data is None:
            for path, _ , files in os.walk(rootfolder):
                for name in files:
                    if name.split('.')[-1] == ext:
                        result = dict()
                        result['parent'] = path
                        self.__paths[os.path.join(path, name)] = result
        else:
            self.__paths = pre_data

        self.__detector = MTCNN(select_largest=True,keep_all=False, device='cuda:0')
        fw, fh = face_expected_ratio.split(':')
        mw, mh = mouth_expected_ratio.split(':')
        self.__faceratiowh = int(fw)/int(fh)
        self.__mouthratiowh = int(mw)/int(mh)
        self.__zoomout_ratio = zoomout_ratio

    def __iterate_frames(self, videofile):
        vidcap = cv2.VideoCapture(videofile)
        while True:
            success, image = vidcap.read()
            if not success:
                return
            if image is None:
                print("image is None")
            yield image

    def __process(self, frame):
        faces = self.__detector.detect(frame, landmarks=True)

        points = faces[2][0]
        mrx, mry = points[4]
        mlx, mly = points[3]
        mw = abs(mrx - mlx) * self.__zoomout_ratio
        mh = mw / self.__mouthratiowh
        mouth_center_x = (mrx + mlx)/2
        mouth_center_y = (mry + mly)/2
        mouthbox = (int(mouth_center_x - mw/2), int(mouth_center_y - mh/2), int(mw), int(mh))

        face = faces[0]
        fx1, fy1, fx2, fy2 = face[0]
        fh = abs(fx1 - fx2)
        fw = abs(fy1 - fy2)

        face_center_x = (fx1 + fx2)/2
        face_center_y = (fy1 + fy2)/2

        if fw/fh > self.__faceratiowh:
            fh = int(fw / self.__faceratiowh * self.__zoomout_ratio)
            fw = int(fw * self.__zoomout_ratio)
        else:
            fh = int(fh * self.__zoomout_ratio)
            fw = int(fh / self.__faceratiowh * self.__zoomout_ratio)

        fx = face_center_x - fw//2
        fy = face_center_y - fh//2

        facebox = (int(fx), int(fy), int(fw), int(fh))

        return facebox, mouthbox


    def __save_result(self, output_path):
        with open(output_path, 'w+') as outfile:
            json.dump(self.__paths, outfile)

    def run(self, output_path):
        cnt = 0
        total = len(self.__paths)
        one_percent = total//100
        print(f"start processing for {total} videos")

        for file, record in self.__paths.items():
            if 'face' in record and len(record['face']) == 4 and 'mouth' in record and len(record['mouth']) == 4:
                continue
            # print(f"processing for {file}")
            for frame in self.__iterate_frames(file):
                try:
                    facebox, mouthbox = self.__process(frame)
                    cnt += 1
                    result = self.__paths[file]
                    result['face'] = facebox
                    result['mouth'] = mouthbox
                    if cnt % one_percent == 0:
                        print(f"{cnt/one_percent}% of videos have been processed!")
                        self.__save_result(output_path)
                    break
                except Exception as e:
                    print(f"exception for {file}: ", e)
                    # traceback.print_exception(*sys.exc_info()) 
            if 'face' not in self.__paths[file]:
                print(f"give up on {file}")
        self.__save_result(output_path)

    def count(self):
        cnt = 0
        for file, record in self.__paths.items():
            if 'face' in record and len(record['face']) == 4 and 'mouth' in record and len(record['mouth']) == 4:
                cnt += 1
        return cnt

def main():
    output_file = "./result.json"
    videos = "/media/tuantran/raid-data/dataset/GRID/video"
    pre_data = None
    if path.exists(output_file):
        with open(output_file, 'r') as infile:
            pre_data = json.load(infile)
    detector = VideoFaceDetector(
            pre_data,
            videos, 
            "mpg",
            "3:4",
            "1:1",
            1.0,
        )
    detector.run(output_file)
    print(detector.count())

if __name__ == "__main__":
    main()
