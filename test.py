from deepface import DeepFace
import cv2 
import os
from retinaface import RetinaFace
import matplotlib.pyplot as plt

TMP_FACES_PATH = "dataset/test_images/tmp"

def int_tuple(t):
    return tuple(int(x) for x in t)

input_path = "dataset/test_images/img1.jpg"
img = cv2.imread(input_path)
resp = RetinaFace.detect_faces(input_path)
i = 0
for key in resp:
    identity = resp[key]
    
    facial_area = identity["facial_area"]

    facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
    # plt.imshow(facial_img[:, :, ::-1])
    # plt.show()
    cv2.imwrite('dataset/test_images/tmp/face_' + str(i) + '.jpg', facial_img)
    i += 1

model_name = "ArcFace"
model = DeepFace.build_model(model_name)
print(model_name," is built")
print("-----------------------------------------")

for face_path in os.listdir(TMP_FACES_PATH):
    print("Face path: ", face_path)
    img_path = os.path.join(TMP_FACES_PATH, face_path)
    img = cv2.imread(img_path)    
    df = DeepFace.find(img, db_path = "dataset/face_db_raw", enforce_detection=False, model=model, model_name=model_name, detector_backend='retinaface')
    print("--------------------------")
    if df.size > 0:
        result = df.iloc[0]
        if result['ArcFace_cosine'] < 0.4:
            identity = result['identity'].split('/')[-1].split('.')[0]
            print("Face identified: ", identity)
        else:
            print("Face not identified")
    else:
        print("No face found")

    print("--------------------------")

    # delete tmp images
    os.remove(img_path)