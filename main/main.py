# Import libraries
import numpy as np
import pandas as pd
import torch

import cv2
import skimage
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import json
import yaml
import logging
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing

def mesure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def ball_assigner(bboxes_p, labels_p, p_ids, ball_bbox_xywh):
    MAX_PLAYER_BALL_DIST = 30 #if above 70px the won't be assinged to anyone
    # get the center of the ball
    ball_position = ball_bbox_xywh[:2]

    minimum_distance = 99999
    assigned_player = -1

    for i in range(bboxes_p.shape[0]):
        # if is player (class 0)
        if labels_p[i]==0:
            player_bbox = bboxes_p[i]
            distance_left = mesure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = mesure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)


            if distance < MAX_PLAYER_BALL_DIST and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = p_ids[i]
    return assigned_player

def process_video(video_path, team_color_dict):
    with open("inputs/pitch map labels position.json", 'r') as f:
        keypoints_map_pos = json.load(f)
    # Get football field keypoints numerical to alphabetical mapping
    with open("inputs/config pitch dataset.yaml", 'r') as file:
        classes_names_dic = yaml.safe_load(file)['names']

    # Get football field keypoints numerical to alphabetical mapping
    with open("inputs/config players dataset.yaml", 'r') as file:
        labels_dic = yaml.safe_load(file)['names']

    # tac_map = cv2.imread('inputs/tactical_map.png')
    nbr_team_colors = 2
    colors_list = team_color_dict[list(team_color_dict.keys())[0]] + team_color_dict[list(team_color_dict.keys())[1]]
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] 
    
    model_players = YOLO("models/Yolo8L Players/weights/best.pt")
    model_keypoints = YOLO("models/Yolo8M Field Keypoints/weights/best.pt")
    model_ball = torch.hub.load("ultralytics/yolov5", "custom", path="models\yolov5l6_trained_600images.pt")
    cap = cv2.VideoCapture(video_path)
    frame_nbr = 0
    keypoints_displacement_mean_tol = 5 #10
    player_model_conf_thresh = 0.60
    keypoints_model_conf_thresh = 0.6 #0.70
    players_teams_list = []

    data_player = {
        "frame_nbr": np.array([]),
        "player_ID": np.array([]),
        "player_pos_X": np.array([]),
        "player_pos_Y": np.array([]),
        "player_team": np.array([]),
        "assigned_id": np.array([])
    }
    data_ball = {
        "frame_nbr": np.array([]),
        "ball_pos_X": np.array([]),
        "ball_pos_Y": np.array([]),
    }
    while cap.isOpened():
    # Update frame counter
        frame_nbr += 1
        print(frame_nbr)
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            results_players = model_players.track(frame, tracker="botsort.yaml", persist=True, conf=player_model_conf_thresh, device="cuda", verbose=False, classes=[0, 1])
            results_keypoints = model_keypoints(frame, conf=keypoints_model_conf_thresh, verbose=False)

            ## Extract detections information
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy().astype(int)              # Detected players, referees and ball (x,y,x,y) bounding boxes
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy().astype(int)            # Detected players, referees and ball (x,y,w,h) bounding boxes    
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     # Detected players, referees and ball labels list
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     # Detected players, referees and ball confidence level
            p_ids = results_players[0].boxes.id.numpy()                                     # Detected players, referees and ball ids
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy().astype(int)            # Detected field keypoints (x,y,w,h) bounding boxes
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy().astype(int)          # Detected field keypoints (x,y,w,h) bounding boxes
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   # Detected field keypoints labels list

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordiantes on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])

            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                # Always calculate homography matrix on the first frame
                if frame_nbr > 1:
                    # Determine common detected field keypoints between previous and current frames
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]   
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]        
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  
                        update_homography = coor_error > keypoints_displacement_mean_tol               
                    else:
                        update_homography = True                                                         
                else:
                    update_homography = True

                if  update_homography:
                    h, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts)                  
                
                detected_labels_prev = detected_labels.copy()                               # Save current detected keypoint labels for next frame
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()               # Save current detected keypoint coordiantes for next frame

                bboxes_p_c_0 = bboxes_p_c[[i==0 for i in labels_p],:]
                results_ball = model_ball(frame, size=1280)
                bboxes_p_c_2 = results_ball.xywh[0][:4].cpu().numpy()        
                bboxes_ball_xyxy = results_ball.xyxy[0][:4].cpu().numpy()     
                if len(bboxes_p_c_2) != 0:
                    high_conf_ball = np.argmax(bboxes_p_c_2[:, 3])
                    bboxes_p_c_2 = bboxes_p_c_2[high_conf_ball]
                    bboxes_ball_xyxy = bboxes_ball_xyxy[high_conf_ball]

                detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
                detected_ball_src_pos = bboxes_p_c_2[:2] if bboxes_p_c_2.shape[0]>0 else None

                pred_dst_pts = []                                                           # Initialize players tactical map coordiantes list
                for pt in detected_ppos_src_pts:                                            # Loop over players frame coordiantes
                    pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Covert to homogeneous coordiantes
                    dest_point = np.matmul(h, np.transpose(pt))                              # Apply homography transofrmation
                    dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordiantes
                    pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
                pred_dst_pts = np.array(pred_dst_pts)

                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(h, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

            ######### Part 2 ########## 
            # Players Team Prediction #
            ###########################
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                      # Convert frame to RGB
            obj_palette_list = []                                                                   # Initialize players color palette list
            palette_interval = (0,5)                                                                # Color interval to extract from dominant colors palette (1rd to 5th color)
            annotated_frame = frame                                                                 # Create annotated frame 

            ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
            for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
                if int(j) == 0:
                    bbox = results_players[0].boxes.xyxy.cpu().numpy()[i,:]                         # Get bbox info (x,y,x,y)
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
                    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                    center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
                    center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
                    center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
                    center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
                    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                                            center_filter_x1:center_filter_x2]
                    obj_pil_img = Image.fromarray(np.uint8(center_filter))                          # Convert to pillow image
                        
                    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
                    palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
                    palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
                    color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
                    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                                        by = 'cnt', ascending = False).iloc[
                                            palette_interval[0]:palette_interval[1],:]
                    palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
                    # Update detected players color palette list
                    obj_palette_list.append(palette)
            
            ## Calculate distances between each color from every detected player color palette and the predefined teams colors
            players_distance_features = []
            # Loop over detected players extracted color palettes
            for palette in obj_palette_list:
                palette_distance = []
                palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
                # Loop over colors in palette
                for color in palette_lab:
                    distance_list = []
                    # Loop over predefined list of teams colors
                    for c in color_list_lab:
                        #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                        distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
                        distance_list.append(distance)                                              # Update distance list for current color
                    palette_distance.append(distance_list)                                          # Update distance list for current palette
                players_distance_features.append(palette_distance)                                  # Update distance features list
            # Loop over players distance features
            for distance_feats in players_distance_features:
                vote_list=[]
                # Loop over distances for each color 
                for dist_list in distance_feats:
                    team_idx = dist_list.index(min(dist_list))//nbr_team_colors                     # Assign team index for current color based on min distance
                    vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
                players_teams_list.append(max(vote_list, key=vote_list.count))                      # Predict current player team by vote counting

            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################
            ball_color_bgr = (0,0,255)                        # Color (GBR) for ball annotation on tactical map
            j=0                                              # Initializing counter of detected players                                
            if len(bboxes_p_c_2) > 0:
                closest_player_id = ball_assigner(bboxes_p, labels_p, p_ids, bboxes_p_c_2)
            for i in range(bboxes_p.shape[0]):
                conf = confs_p[i]                                                                               # Get confidence of current detected object
                if labels_p[i]==0:                                                                              # Display annotation for detected players (label 0)

                    team_name = list(team_color_dict.keys())[players_teams_list[j]]                               # Get detected player team prediction

                    if len(detected_labels_src_pts)>3:
                        data_player["frame_nbr"] = np.append(data_player["frame_nbr"], frame_nbr)
                        data_player["player_ID"] = np.append(data_player["player_ID"], p_ids[i])
                        data_player["player_pos_X"] = np.append(data_player["player_pos_X"], pred_dst_pts[j][0])
                        data_player["player_pos_Y"] = np.append(data_player["player_pos_Y"], pred_dst_pts[j][1])
                        data_player["player_team"] = np.append(data_player["player_team"], team_name)
                        data_player["assigned_id"] = np.append(data_player["assigned_id"], closest_player_id)
                    else:
                        data_player["frame_nbr"] = np.append(data_player["frame_nbr"], frame_nbr)
                        data_player["player_ID"] = np.append(data_player["player_ID"], p_ids[i])
                        data_player["player_pos_X"] = np.append(data_player["player_pos_X"], np.nan)
                        data_player["player_pos_Y"] = np.append(data_player["player_pos_Y"], np.nan)
                        data_player["player_team"] = np.append(data_player["player_team"], team_name)
                        data_player["assigned_id"] = np.append(data_player["assigned_id"], closest_player_id) 
                    ###ADDED###
                    j+=1
                    # Update players counter
            # Add tactical map ball postion annotation if detected
            if detected_ball_src_pos is not None:
                data_ball["frame_nbr"] = np.append(data_ball["frame_nbr"], frame_nbr)
                data_ball["ball_pos_X"] = np.append(data_ball["ball_pos_X"], detected_ball_dst_pos[0])
                data_ball["ball_pos_Y"] = np.append(data_ball["ball_pos_Y"], detected_ball_dst_pos[1])
            else:
                data_ball["frame_nbr"] = np.append(data_ball["frame_nbr"], frame_nbr)
                data_ball["ball_pos_X"] = np.append(data_ball["ball_pos_X"], np.nan)
                data_ball["ball_pos_Y"] = np.append(data_ball["ball_pos_Y"], np.nan)
            if frame_nbr == 100:
                break
        else:
            break
    cap.release()