import cv2
import numpy as np 
import os
import random 
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device = device)

save_folder = "/project/msoleyma_1026/personality_detection/first_impressions_v2_dataset/testing/video_feature_vectors"
video_dir = "/project/msoleyma_1026/personality_detection/first_impressions_v2_dataset/testing/video_files"

training=True 

def get_image_embedding(img):
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    pil_image = Image.fromarray(img)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim = -1, keepdim = True)
    # print("embedding shape = ", image_features.shape) 
    return image_features

def get_number_of_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return total_frames
    
def extract_N_video_frames(file_path, number_of_samples = 5):
    nb_frames = int(get_number_of_frames(file_path))
    
    video_frames = []
    # random_indexes = random.sample(range(0, nb_frames), number_of_samples)
    frame_indices = np.linspace(0, nb_frames - 1, number_of_samples, dtype=int)
    # print("file path = ", file_path, "num frames = ", nb_frames, "frame indices = ", frame_indices)
    
    cap = cv2.VideoCapture(file_path)
    for ind in frame_indices:
        # cap.set(1,ind)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        res, frame = cap.read()
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    del cap
    return video_frames

def resize_image(image, new_size):
    return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def crop_image_window(image: np.ndarray, training=True):
    height, width, _ = image.shape
    if training:
        MAX_N = height - 128
        MAX_M = width - 128
        rand_N_index, rand_M_index = random.randint(0, MAX_N) , random.randint(0, MAX_M)
        return np.array(image[rand_N_index:(rand_N_index+128),rand_M_index:(rand_M_index+128),:])
    else:
        N_index = (height - 128) // 2
        M_index = (width - 128) // 2
        return np.array(image[N_index:(N_index+128),M_index:(M_index+128),:])

for vid_fold in os.listdir(video_dir):
    if not vid_fold.startswith("test-"): 
        continue 
    print(vid_fold)
    vid_fold_path = os.path.join(video_dir, vid_fold) 
    vid_feat_fold = os.path.join(save_folder, vid_fold) 
    if not os.path.isdir(vid_feat_fold): 
        os.makedirs(vid_feat_fold)
    for vid_file_fold in os.listdir(vid_fold_path): 
        print( vid_file_fold)
        video_folder = os.path.join(vid_fold_path, vid_file_fold)
        feature_set = set(os.listdir(vid_feat_fold))

        for vid_file in os.listdir(video_folder):
            video_file = os.path.join(vid_fold_path, vid_file_fold, vid_file) 
            feature_file = vid_file[:-4]+".npy"
            if feature_file in feature_set: 
                continue
            output_file = os.path.join(vid_feat_fold, feature_file)
            
            sampled = extract_N_video_frames(file_path= video_file, number_of_samples= 5)
            resized_images = [resize_image(image= im, new_size= (248,140)) for im in sampled]
            cropped_images = [crop_image_window(image= resi,training= training) / 255.0 for resi in resized_images]
            preprocessed_video = np.stack(cropped_images)
            # print(type(preprocessed_video))
            video_feature_vecs = [] 
            for img in preprocessed_video: 
                # print(img.shape, type(img)) 
                # print(img)
                feature_vec = get_image_embedding(img)
                # print(type(feature_vec))
                video_feature_vecs.append(feature_vec.cpu().numpy())
            video_feature_vecs = np.array(video_feature_vecs)
            
            np.save(output_file, video_feature_vecs)

    
