import cv2


def face_detect(image):
    face_detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    # 灰度处理，降低计算量
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''
    输入：
     Image --> 输入图像
     ScaleFactor --> 放缩比率, default为1.1
     minNeighbors --> 表示最低相邻矩形框, default为3
     minSize --> 可以检测的最小人脸
     maxSize --> 可以检测的最大人脸
    输出：
     face --> 人脸的位置 (x, y, w, h)
    '''
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

    # for x, y, width, height in faces:
    #     cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3, cv2.LINE_8, 0)
    # cv2.imshow('face', image)
    # cv2.imwrite('result/face/face.jpg', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    eye_detector = cv2.CascadeClassifier('xml/haarcascade_eye.xml')
    # 基于之前的人脸检测
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3, cv2.LINE_8, 0)

        roi = image[y:y+height, x:x+width]
        eyes = eye_detector.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=2, minSize=(150, 150))

        for ex, ey, ewidth, eheight in eyes:
            cv2.rectangle(roi, (ex, ey), (ex + ewidth, ey + eheight), (255, 0, 0), 3, cv2.LINE_8, 0)

    cv2.imshow('face', image)
    cv2.imwrite('result/face/face.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = cv2.imread('data/frame/frame1.jpg')
    face_detect(image_path)
