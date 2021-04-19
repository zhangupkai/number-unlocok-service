import cv2


def video2frame(videos_path, frames_save_path, time_interval):
    """
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    """
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
        # if count == 20:
        #   break
    print(count)


if __name__ == '__main__':
    videos_path = 'data/video/2021-04-16_14-51-23.mp4'
    frames_save_path = 'data/frame'
    time_interval = 1  # 一帧保存一次
    video2frame(videos_path, frames_save_path, time_interval)
