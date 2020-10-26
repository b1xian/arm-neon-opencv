import os
import cv2


def main():
    video_path = '/Users/v_guojinlong/Desktop/SNPE/VisionVideo'
    video_names = ['close_eye1.mov', 'close_eye2.mov', 'close_eye3.mov']

    save_img_path = os.path.join(video_path, 'images')
    os.system('rm -r ' + save_img_path)
    os.mkdir(save_img_path)

    for video_name in video_names:
        cap = cv2.VideoCapture(os.path.join(video_path, video_name))
        if cap.isOpened():
            print("open video %s success" % video_name)
        else:
            print("open video %s failed" % video_name)
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame count:%d" % frame_count)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("width :%d, height:%d" % (width, height))
        i = 0
        while cap.isOpened():
            flag, frame = cap.read()
            if flag:
                i += 1
                print(i)
                cv2.imwrite(os.path.join(save_img_path, video_name+'_'+str(i)+'.jpg'), frame)
            else:
                break
        cap.release()

if __name__ == '__main__':
    main()