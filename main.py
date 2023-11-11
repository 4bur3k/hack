import streamlit as st
import numpy as np
import cv2

from ultralytics import YOLO

w, h = 1920, 1080

link = ''
file = None
model = None
# login, password = 'admin', 'A1234567'

model = YOLO('best.pt')

with st.sidebar:
    mode = st.radio('Формат обработки', ['Online', 'Ofline'])

    if mode =='Online':
        link = st.text_input('rtsp адрес')
        login = st.text_input('Логин')
        password = st.text_input('Пароль')
    elif mode == 'Ofline':
        file = st.file_uploader('Load video')
        print('file uploaded')

# VIEWER_WIDTH = 640
def get_random_numpy():

    return np.random.randint(0, 100, size=(32, 32))

print(mode, link, file)
if mode == 'Online' and link != '':
    print('online mode')
    viewer = st.image(get_random_numpy())

    if '.mp4' in link:
        new_link=link
    else:
        sign=':'
        l_p=''.join([login, 
                     sign, password
                     ])
        
        start_position = link.index("rtsp://") + len("rtsp://")
        end_position = link.index("@")
        new_link = link.replace(link[start_position:end_position], l_p)

    cap = cv2.VideoCapture(new_link) 

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(cv2.resize(frame, (w,h)), cv2.COLOR_BGR2RGB)
            results = model.track(frame, conf=0.3, iou=0.5)
            print(results)
            annotated_frame = results[0].plot()
            # cv2.imshow('Frame',frame)
            viewer.image(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif mode == 'Ofline' and file!= None:
    print('Ofline mode')

    bytes_data = file.getvalue()
    with open('tmp.mp4', 'wb') as f:
        f.write(bytes_data)

    vid = model.track('tmp.mp4', conf=0.3, iou=0.5)

    st.video(vid)
