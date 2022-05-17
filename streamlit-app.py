import streamlit as st
import cv2
import os
from PIL import Image
from retinaface import RetinaFace
from deepface import DeepFace

TMP_FACES_PATH = "dataset/test_images/tmp"
DB_PATH = "dataset/face_db_raw"
UPLOADED_IMG_PATH = "dataset/test_images/uploaded_image.jpg"

def load_image(image_file):
	img = Image.open(image_file)
	return img

def int_tuple(t):
    return tuple(int(x) for x in t)

st.set_page_config(layout="wide")

st.header("BIT - Face Recognition")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:   
        with open(UPLOADED_IMG_PATH,"wb") as f: 
            f.write(image_file.getbuffer())     
        img = cv2.imread(UPLOADED_IMG_PATH)
        
        resp = RetinaFace.detect_faces(UPLOADED_IMG_PATH)
        i = 0
        for key in resp:
            identity = resp[key]
            rectangle_color = (255, 255, 255)

            landmarks = identity["landmarks"]
            diameter = 1
            cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
            cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
            cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
            cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), -1)
            cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)

            facial_area = identity["facial_area"]
            
            cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), rectangle_color, 1)

            facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
            # plt.imshow(facial_img[:, :, ::-1])
            # plt.show()
            cv2.imwrite(TMP_FACES_PATH + '/face_' + str(i) + '.jpg', facial_img)
            i += 1
        os.remove(UPLOADED_IMG_PATH)

        model_name = "ArcFace"
        model = DeepFace.build_model(model_name)

        for face_path in os.listdir(TMP_FACES_PATH):
            print("Face path: ", face_path)
            img_path = os.path.join(TMP_FACES_PATH, face_path)
            img_face = cv2.imread(img_path)    
            df = DeepFace.find(img_face, db_path = DB_PATH, enforce_detection=False, model=model, model_name=model_name, detector_backend='retinaface')
            print("--------------------------")
            if df.size > 0:
                result = df.iloc[0]
                if result['ArcFace_cosine'] < 0.4:
                    identity = result['identity'].split('/')[-1].split('.')[0]
                    st.write("Face identified: ", identity)
                else:
                    st.write("Face not identified")
            else:
                st.write("No face found")

            print("--------------------------")
        
            # delete tmp images
            os.remove(img_path)
        st.write("--------------------------")
with col2:
    if image_file is not None:
        # To View Uploaded Image
        # st.image(load_image(image_file))
        st.image(img[:, :, ::-1])
        
    else:
        st.write('No file selected')