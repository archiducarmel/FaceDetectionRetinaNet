from fdet import io, MTCNN, RetinaFace
from fdet.utils.io import VideoHandle
import cv2, time

#video = VideoHandle('0')
#detector = MTCNN()
detector = RetinaFace(backbone='RESNET50')
img = "imgs/Africa_8.jpg"
image = io.read_as_rgb(img)

video_source = "videos/vid.mp4"
video_source = "/home/sitou/Vidéos/videos_covid/test-covid1.mp4"
video_source = "/home/sitou/Vidéos/videos_covid/zi_pf1_sb1-1-distanciation.avi"

cap = cv2.VideoCapture(video_source)

while True:
    t0 = time.time()
    _, image = cap.read()
    detections = detector.detect(image)
    print("Inference time : {0} msec". format(int((time.time()-t0)*1000)))
    #print(detections)

    output_image = io.draw_detections(image, detections, color='white', thickness=5)
    cv2.namedWindow("FACE DETECTION WITH RETINAFACE", cv2.WINDOW_NORMAL)
    cv2.imshow("FACE DETECTION WITH RETINAFACE", output_image)
    cv2.waitKey(20)
    #io.save('output.jpg', output_image)
