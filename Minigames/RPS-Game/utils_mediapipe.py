###############################################################################
### Wrapper for Google MediaPipe face, hand, body, holistic and object pose estimation
### https://github.com/google/mediapipe
###############################################################################

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Resolve model .task files relative to THIS file, not the caller's working
# directory — so the game works whether launched from its own folder or from
# the master hub at the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _model(name):
    return os.path.join(_HERE, name)


# Define default camera intrinsic
img_width  = 640
img_height = 480
intrin_default = {
    'fx': img_width*0.9, # Approx 0.7w < f < w https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    'fy': img_width*0.9,
    'cx': img_width*0.5, # Approx center of image
    'cy': img_height*0.5,
    'width': img_width,
    'height': img_height,
}


class MediaPipeFace:
    def __init__(self, static_image_mode=True, max_num_faces=1):
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        self._running_mode = running_mode
        self._timestamp_ms = 0

        base_options = mp_tasks.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.pipe = vision.FaceLandmarker.create_from_options(options)

        # Define face parameter
        # Note: FaceLandmarker v2 returns 478 landmarks (468 face + 10 iris)
        self.param = []
        for _ in range(max_num_faces):
            p = {
                'detect'  : False, # Boolean to indicate whether a face is detected
                'keypt'   : np.zeros((478,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((478,3)), # 3D joint in relative coordinate
                'fps'     : -1, # Frame per sec
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['detect'] = False

        if result.face_landmarks:
            for i, face_landmarks in enumerate(result.face_landmarks):
                self.param[i]['detect'] = True
                for j, lm in enumerate(face_landmarks):
                    self.param[i]['keypt'][j,0] = lm.x * img_width
                    self.param[i]['keypt'][j,1] = lm.y * img_height
                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

        return self.param


    def forward(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        if self._running_mode == vision.RunningMode.IMAGE:
            result = self.pipe.detect(mp_image)
        else:
            self._timestamp_ms += 1
            result = self.pipe.detect_for_video(mp_image, self._timestamp_ms)
        return self.result_to_param(result, img)


class MediaPipeHand:
    def __init__(self, static_image_mode=True, max_num_hands=1, intrin=None):
        self.max_num_hands = max_num_hands
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        self._running_mode = running_mode
        self._timestamp_ms = 0

        base_options = mp_tasks.BaseOptions(model_asset_path=_model('hand_landmarker.task'))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.pipe = vision.HandLandmarker.create_from_options(options)

        # Define hand parameter
        self.param = []
        for _ in range(max_num_hands):
            p = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
                # https://github.com/google/mediapipe/issues/1351
                # 'visible' : np.zeros(21), # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                # 'presence': np.zeros(21), # Presence: Likelihood [0,1] of being present in the image or if its located outside the image
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['class'] = None

        if result.hand_landmarks:
            for i, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                if i > self.max_num_hands - 1: break
                self.param[i]['class'] = handedness[0].category_name
                self.param[i]['score'] = handedness[0].score

                for j, lm in enumerate(hand_landmarks):
                    self.param[i]['keypt'][j,0] = lm.x * img_width
                    self.param[i]['keypt'][j,1] = lm.y * img_height
                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

                self.param[i]['angle'] = self.convert_3d_joint_to_angle(self.param[i]['joint'])
                self.convert_relative_to_actual_3d_joint(self.param[i], self.intrin)

        return self.param

    
    def convert_3d_joint_to_angle(self, joint):
        # Get direction vector of bone from parent to child
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
        v = v2 - v1 # [20,3]
        # Normalize v
        v = v/np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        return np.degrees(angle) # Convert radian to degree


    def convert_relative_to_actual_3d_joint(self, param, intrin):
        # Note: MediaPipe hand model uses weak perspective (scaled orthographic) projection
        # https://github.com/google/mediapipe/issues/742#issuecomment-639104199

        # Weak perspective projection = (X,Y,Z) -> (X,Y) -> (SX, SY) = (x,y) in image coor
        # https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect5.pdf (slide 35) 
        # Step 1) Orthographic projection = (X,Y,Z) -> (X,Y) discard Z depth
        # Step 2) Uniform scaling by a factor S = f/Zavg, (X,Y) -> (SX, SY)
        # Therefore, to backproject 2D -> 3D:
        # x = SX + cx -> X = (x - cx) / S
        # y = SY + cy -> Y = (y - cy) / S
        # z = SZ      -> Z = z / S

        # Note: Output of mediapipe 3D hand joint X' and Y' are normalized to [0,1]
        # Need to convert normalized 3D (X',Y') to 2D image coor (x,y)
        # x = X' * img_width
        # y = Y' * img_height

        # Note: For scaling of mediapipe 3D hand joint Z'
        # Since it is mentioned in mcclanahoochie's comment to the above github issue
        # 'z is scaled proportionally along with x and y (via weak projection), and expressed in the same units as x & y.'
        # And also in the paper for MediaPipe face: 2019 Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs
        # '3D positions are re-scaled so that a fixed aspect ratio is maintained between the span of x-coor and the span of z-coor'
        # Therefore, I think that Z' is scaled similar to X'
        # z = Z' * img_width
        # z = SZ -> Z = z/S

        # Note: For full-body pose the magnitude of z uses roughly the same scale as x
        # https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
        
        # De-normalized 3D hand joint
        param['joint_3d'][:,0] = param['joint'][:,0]*intrin['width'] -intrin['cx']
        param['joint_3d'][:,1] = param['joint'][:,1]*intrin['height']-intrin['cy']
        param['joint_3d'][:,2] = param['joint'][:,2]*intrin['width']

        # Assume average depth is fixed at 0.6 m (works best when the hand is around 0.5 to 0.7 m from camera)
        Zavg = 0.6
        # Average focal length of fx and fy
        favg = (intrin['fx']+intrin['fy'])*0.5
        # Compute scaling factor S
        S = favg/Zavg
        # Uniform scaling
        param['joint_3d'] /= S

        # Estimate wrist depth using similar triangle
        D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
        # Dist btw wrist and index finger MCP keypt (in 2D image coor)
        d = np.linalg.norm(param['keypt'][0] - param['keypt'][9])
        # d/f = D/Z -> Z = D/d*f
        Zwrist = D/d*favg
        # Add wrist depth to all joints
        param['joint_3d'][:,2] += Zwrist      


    def forward(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        if self._running_mode == vision.RunningMode.IMAGE:
            result = self.pipe.detect(mp_image)
        else:
            self._timestamp_ms += 1
            result = self.pipe.detect_for_video(mp_image, self._timestamp_ms)
        return self.result_to_param(result, img)


class MediaPipeBody:
    def __init__(self, static_image_mode=True, model_complexity=1, intrin=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # model_complexity 0/1/2 maps to lite/full/heavy model files
        model_files = {0: 'pose_landmarker_lite.task', 1: 'pose_landmarker_full.task', 2: 'pose_landmarker_heavy.task'}
        model_file = model_files.get(model_complexity, 'pose_landmarker_full.task')

        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        self._running_mode = running_mode
        self._timestamp_ms = 0

        base_options = mp_tasks.BaseOptions(model_asset_path=model_file)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.pipe = vision.PoseLandmarker.create_from_options(options)

        # Define body parameter
        self.param = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((33,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((33,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((33,3)), # 3D joint in camera coordinate (m)
                'visible' : np.zeros(33),     # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        img_height, img_width, _ = img.shape

        if not result.pose_landmarks:
            self.param['detect'] = False
        else:
            self.param['detect'] = True

            for j, lm in enumerate(result.pose_landmarks[0]):
                self.param['keypt'][j,0] = lm.x * img_width
                self.param['keypt'][j,1] = lm.y * img_height
                self.param['joint'][j,0] = lm.x
                self.param['joint'][j,1] = lm.y
                self.param['joint'][j,2] = lm.z
                self.param['visible'][j] = lm.visibility

            self.convert_relative_to_actual_3d_joint(self.param, self.intrin)

        return self.param


    def convert_relative_to_actual_3d_joint(self, param, intrin):
        # De-normalized 3D body joint
        param['joint_3d'][:,0] = param['joint'][:,0]*intrin['width'] -intrin['cx']
        param['joint_3d'][:,1] = param['joint'][:,1]*intrin['height']-intrin['cy']
        param['joint_3d'][:,2] = param['joint'][:,2]*intrin['width'] * 0.25
        # Note: Seems like need to further scale down z by 0.25 else will get elongated forearm and feet
        # Could it be beacuse z ranges btw a smaller range of -ve XX to +ve XX unlike x and y which range from 0 to 1

        # Compute center of shoulder and hip joint
        center_shoulder = (param['joint_3d'][11] + param['joint_3d'][12])*0.5
        center_hip      = (param['joint_3d'][23] + param['joint_3d'][24])*0.5

        # Translate to new origin at center of hip
        param['joint_3d'] -= center_hip

        # Scale from relative to actual 3D joint in m
        D = 0.55 # Note: Hardcode actual dist btw shoulder and hip joint as 0.55 m
        d = np.linalg.norm(center_shoulder - center_hip) # Dist btw shoulder and hip joint (in relative 3D coor)
        param['joint_3d'] *= D/d

        # Note: Unlike hand where the Zavg is relatively constant around 0.6 m, 
        # it is quite hard to define Zavg for body, 
        # thus the step to convert to camera coor is ignored


    def forward(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        if self._running_mode == vision.RunningMode.IMAGE:
            result = self.pipe.detect(mp_image)
        else:
            self._timestamp_ms += 1
            result = self.pipe.detect_for_video(mp_image, self._timestamp_ms)
        return self.result_to_param(result, img)


class MediaPipeHolistic:
    def __init__(self, static_image_mode=True, model_complexity=1, intrin=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        self._running_mode = running_mode
        self._timestamp_ms = 0

        # Note: model_complexity and smooth_landmarks are not available in HolisticLandmarkerOptions
        base_options = mp_tasks.BaseOptions(model_asset_path='holistic_landmarker.task')
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5)
        self.pipe = vision.HolisticLandmarker.create_from_options(options)

        # Define face parameter
        self.param_fc = {
                'detect'  : False, # Boolean to indicate whether a face is detected
                'keypt'   : np.zeros((468,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((468,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((468,3)), # 3D joint in camera coordinate (m)
                'fps'     : -1, # Frame per sec
            }

        # Define left and right hand parameter
        self.param_lh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }
        self.param_rh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }

        # Define body parameter
        self.param_bd = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((33,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((33,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((33,3)), # 3D joint in camera coordinate (m)
                'visible' : np.zeros(33),     # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        img_height, img_width, _ = img.shape

        ############
        ### Face ###
        ############
        # HolisticLandmarkerResult.face_landmarks is a flat List[NormalizedLandmark], not NormalizedLandmarkList
        if not result.face_landmarks:
            self.param_fc['detect'] = False
        else:
            self.param_fc['detect'] = True
            for j, lm in enumerate(result.face_landmarks):
                self.param_fc['keypt'][j,0] = lm.x * img_width
                self.param_fc['keypt'][j,1] = lm.y * img_height
                self.param_fc['joint'][j,0] = lm.x
                self.param_fc['joint'][j,1] = lm.y
                self.param_fc['joint'][j,2] = lm.z

        #################
        ### Left Hand ###
        #################
        if not result.left_hand_landmarks:
            self.param_lh['class'] = None
        else:
            self.param_lh['class'] = 'left'
            for j, lm in enumerate(result.left_hand_landmarks):
                self.param_lh['keypt'][j,0] = lm.x * img_width
                self.param_lh['keypt'][j,1] = lm.y * img_height
                self.param_lh['joint'][j,0] = lm.x
                self.param_lh['joint'][j,1] = lm.y
                self.param_lh['joint'][j,2] = lm.z
            self.param_lh['angle'] = self.convert_3d_joint_to_angle(self.param_lh['joint'])

        ##################
        ### Right Hand ###
        ##################
        if not result.right_hand_landmarks:
            self.param_rh['class'] = None
        else:
            self.param_rh['class'] = 'right'
            for j, lm in enumerate(result.right_hand_landmarks):
                self.param_rh['keypt'][j,0] = lm.x * img_width
                self.param_rh['keypt'][j,1] = lm.y * img_height
                self.param_rh['joint'][j,0] = lm.x
                self.param_rh['joint'][j,1] = lm.y
                self.param_rh['joint'][j,2] = lm.z
            self.param_rh['angle'] = self.convert_3d_joint_to_angle(self.param_rh['joint'])

        ############
        ### Pose ###
        ############
        if not result.pose_landmarks:
            self.param_bd['detect'] = False
        else:
            self.param_bd['detect'] = True
            for j, lm in enumerate(result.pose_landmarks):
                self.param_bd['keypt'][j,0] = lm.x * img_width
                self.param_bd['keypt'][j,1] = lm.y * img_height
                self.param_bd['joint'][j,0] = lm.x
                self.param_bd['joint'][j,1] = lm.y
                self.param_bd['joint'][j,2] = lm.z
                self.param_bd['visible'][j] = lm.visibility
            self.convert_relative_to_actual_3d_joint(
                self.param_fc, self.param_lh, self.param_rh, self.param_bd, self.intrin)

        return (self.param_fc, self.param_lh, self.param_rh, self.param_bd)

    
    def convert_3d_joint_to_angle(self, joint):
        # Get direction vector of bone from parent to child
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
        v = v2 - v1 # [20,3]
        # Normalize v
        v = v/np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        return np.degrees(angle) # Convert radian to degree


    def convert_relative_to_actual_3d_joint(self, param_fc, param_lh, param_rh, param_bd, intrin):
        if param_bd['detect']:
            # De-normalized 3D body joint
            param_bd['joint_3d'][:,0] = param_bd['joint'][:,0]*intrin['width'] -intrin['cx']
            param_bd['joint_3d'][:,1] = param_bd['joint'][:,1]*intrin['height']-intrin['cy']
            param_bd['joint_3d'][:,2] = param_bd['joint'][:,2]*intrin['width'] * 0.25
            # Note: Seems like need to further scale down z by 0.25 else will get elongated forearm and feet
            # Could it be beacuse z ranges btw a smaller range of -ve XX to +ve XX unlike x and y which range from 0 to 1

            # Compute center of shoulder and hip joint
            center_shoulder = (param_bd['joint_3d'][11] + param_bd['joint_3d'][12])*0.5
            center_hip      = (param_bd['joint_3d'][23] + param_bd['joint_3d'][24])*0.5

            # Translate to new origin at center of hip
            param_bd['joint_3d'] -= center_hip

            # Scale from relative to actual 3D joint in m
            D = 0.55 # Note: Hardcode actual dist btw shoulder and hip joint as 0.55 m
            d = np.linalg.norm(center_shoulder - center_hip) # Dist btw shoulder and hip joint (in relative 3D coor)
            param_bd['joint_3d'] *= D/d

        if param_fc['detect']:
            param_fc['joint_3d'] = param_fc['joint'].copy()
            
            # Scale from relative to actual 3D joint in m
            D = 0.07 # Note: Hardcode actual dist btw left and right eye as 0.07 m
            d = np.linalg.norm(param_fc['joint_3d'][386] - param_fc['joint_3d'][159]) # Dist btw left and right eye (in relative 3D coor)
            param_fc['joint_3d'] *= D/d

            # Translate to face nose joint then add body nose joint
            param_fc['joint_3d'] += -param_fc['joint_3d'][4] + param_bd['joint_3d'][0] # Nose joint             

        if param_lh['class'] is not None:
            # De-normalized 3D hand joint
            param_lh['joint_3d'][:,0] = param_lh['joint'][:,0]*intrin['width'] -intrin['cx']
            param_lh['joint_3d'][:,1] = param_lh['joint'][:,1]*intrin['height']-intrin['cy']
            param_lh['joint_3d'][:,2] = param_lh['joint'][:,2]*intrin['width']

            # Scale from relative to actual 3D joint in m
            D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
            d = np.linalg.norm(param_lh['joint_3d'][0] - param_lh['joint_3d'][9]) # Dist btw wrist and index finger MCP joint
            param_lh['joint_3d'] *= D/d

            # Translate to original hand wrist then add body wrist joint
            param_lh['joint_3d'] += -param_lh['joint_3d'][0] + param_bd['joint_3d'][15] # Left wrist joint

        if param_rh['class'] is not None:
            # De-normalized 3D hand joint
            param_rh['joint_3d'][:,0] = param_rh['joint'][:,0]*intrin['width'] -intrin['cx']
            param_rh['joint_3d'][:,1] = param_rh['joint'][:,1]*intrin['height']-intrin['cy']
            param_rh['joint_3d'][:,2] = param_rh['joint'][:,2]*intrin['width']
            
            # Scale from relative to actual 3D joint in m
            D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
            d = np.linalg.norm(param_rh['joint_3d'][0] - param_rh['joint_3d'][9]) # Dist btw wrist and index finger MCP joint
            param_rh['joint_3d'] *= D/d

            # Translate to original hand wrist then add body wrist joint
            param_rh['joint_3d'] += -param_rh['joint_3d'][0] + param_bd['joint_3d'][16] # Right wrist joint


    def forward(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        if self._running_mode == vision.RunningMode.IMAGE:
            result = self.pipe.detect(mp_image)
        else:
            self._timestamp_ms += 1
            result = self.pipe.detect_for_video(mp_image, self._timestamp_ms)
        return self.result_to_param(result, img)


class MediaPipeObjectron:
    def __init__(self, *_, **__):
        raise NotImplementedError(
            "MediaPipeObjectron is not available in MediaPipe 0.10+. "
            "The Objectron solution was removed and has no Tasks API replacement."
        )