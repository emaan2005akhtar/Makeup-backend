import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# ================= LANDMARKS =================

LIPS = [61,185,40,39,37,0,267,269,270,409,
        291,375,321,405,314,17,84,181,91,146]

LEFT_BLUSH = [116,123,147,187,205,50]
RIGHT_BLUSH = [345,352,376,411,425,280]


# ================= HELPERS =================

def get_polygon(landmarks, idx, w, h):

    pts=[]

    for i in idx:
        pts.append([
            int(landmarks[i].x*w),
            int(landmarks[i].y*h)
        ])

    return np.array(pts,np.int32)


def create_soft_mask(shape, pts, blur=51):

    mask=np.zeros(shape[:2],dtype=np.uint8)

    cv2.fillPoly(mask,[pts],255)

    mask=cv2.GaussianBlur(mask,(blur,blur),blur//2)/255.0

    return np.expand_dims(mask,2)


# ================= LIP COLOR EXTRACTION =================

def extract_lip_color(image, mask):

    pixels=image[mask>0]

    if len(pixels)<20:
        return np.array([0,0,0],dtype=np.uint8)

    hsv=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV)

    hsv=hsv.reshape(-1,3)

    hsv=hsv[hsv[:,1]>80]

    if len(hsv)==0:
        hsv=pixels

    kmeans=KMeans(n_clusters=3,n_init=10)

    kmeans.fit(hsv)

    counts=np.bincount(kmeans.labels_)

    dominant=kmeans.cluster_centers_[np.argmax(counts)]

    color=cv2.cvtColor(np.uint8([[dominant]]),cv2.COLOR_HSV2BGR)[0][0]

    return color


# ================= BLEND FUNCTION =================

def overlay_blend(base,color,mask,intensity):

    base = base.astype(np.float32)

    color_layer = np.full_like(base,color,dtype=np.float32)

    result = base*(1-mask*intensity) + color_layer*(mask*intensity)

    result = np.clip(result,0,255)

    return result.astype(np.uint8)

# ================= BLUSH GRADIENT =================

def create_blush_gradient(shape, center_x, center_y, radius):

    h,w = shape[:2]

    mask = np.zeros((h,w), dtype=np.float32)

    for y in range(h):
        for x in range(w):

            dist = np.sqrt((x-center_x)**2 + (y-center_y)**2)

            value = max(0, 1 - dist/radius)

            mask[y,x] = value

    mask = cv2.GaussianBlur(mask,(101,101),50)

    return np.expand_dims(mask,2)


# ================= MAIN BLEND =================

def blend_makeup(original_path, ai_path, reference_path):

    original=cv2.imread(original_path)
    ai=cv2.imread(ai_path)
    ref=cv2.imread(reference_path)

    h,w=original.shape[:2]

    ai=cv2.resize(ai,(w,h))

    result=ai.copy()

    rgb=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
    ref_rgb=cv2.cvtColor(ref,cv2.COLOR_BGR2RGB)

    user_results=face_mesh.process(rgb)
    ref_results=face_mesh.process(ref_rgb)

    if not user_results.multi_face_landmarks:
        return ai_path

    user_landmarks=user_results.multi_face_landmarks[0].landmark
    ref_landmarks=ref_results.multi_face_landmarks[0].landmark

    rh,rw,_=ref.shape


    # ================= LIPSTICK =================

    ref_lip_mask=np.zeros((rh,rw),dtype=np.uint8)

    ref_pts=get_polygon(ref_landmarks,LIPS,rw,rh)

    cv2.fillPoly(ref_lip_mask,[ref_pts],255)

    lip_color=extract_lip_color(ref,ref_lip_mask)

    user_lip_mask=create_soft_mask(
        result.shape,
        get_polygon(user_landmarks,LIPS,w,h),
        15
    )

    result=overlay_blend(result,lip_color,user_lip_mask,1.0)


    # ================= BLUSH =================

    ref_blush_mask=np.zeros((rh,rw),dtype=np.uint8)

    cv2.fillPoly(ref_blush_mask,
        [get_polygon(ref_landmarks,LEFT_BLUSH,rw,rh)],255)

    cv2.fillPoly(ref_blush_mask,
        [get_polygon(ref_landmarks,RIGHT_BLUSH,rw,rh)],255)

    blush_color=extract_lip_color(ref,ref_blush_mask)

    left_cheek=get_polygon(user_landmarks,LEFT_BLUSH,w,h).mean(axis=0).astype(int)
    right_cheek=get_polygon(user_landmarks,RIGHT_BLUSH,w,h).mean(axis=0).astype(int)

    left_mask=create_blush_gradient(result.shape,left_cheek[0],left_cheek[1],120)
    right_mask=create_blush_gradient(result.shape,right_cheek[0],right_cheek[1],120)

    blush_mask=left_mask+right_mask

    result=overlay_blend(result,blush_color,blush_mask,1.1)


    # ================= SAVE RESULT =================

    output_path="outputs/result.jpg"

    cv2.imwrite(output_path,result)

    return output_path