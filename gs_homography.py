##imports
import json
import argparse
import sys
from jsonschema import ValidationError
import os
import cv2
import numpy as np
import torch
from glob import glob
import shutil

def _run(command_args):
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "py",
            metavar="<python file>",
            type=str,
       )
    parser.add_argument(
            "filepath",
            metavar="<json_config>",
            type=str,
            help="json config file. Use '-' to accept standard input or pipe.",
        )
    args = parser.parse_args(command_args)
    try:
            if args.filepath == "-":
                if sys.stdin.isatty():
                    print(
                        "Cannot read config from raw 'stdin'; must pipe or redirect a file"
                    )
                    return 1
                print("Reading config from stdin...")
                config = json.load(open(args.filepath))
            else:
                config = json.load(open(args.filepath))
    except ValidationError as e:
            print(
                f"Could not validate config: {e.message} @ {'.'.join(e.absolute_path)}"
            )
            return 1
    except json.decoder.JSONDecodeError:
            if args.filepath == "-":
                print("'stdin' did not provide a json-parsable input")
            else:
                print(f"Could not decode '{args.filepath}' as a json file.")
                if not args.filepath.lower().endswith(".json"):
                    print(f"{args.filepath} is not a '*.json' file")
            return 1
    
    #print parameters being used for calculating homography
    print(config["homography"])

    # create a copy of the old data to modify
    shutil.copytree(config["old_data_path"], config["new_data_path"])

    # Align the green screen
    ags = AlignGreenScreen(**config["homography"])
    
    if config["infer_from_first_frame"] =="no":
        ## If reference images are given for all videos manually
        for folder in config["data"].keys():
            ags.get_gscoords_for_video(folder, folder + "/rgb/" + config["data"][folder]+".png")
    
    elif config["infer_from_first_frame"] =="yes":
        ## if reference is fixed (1st image has all 4 corners visible ) 
        folders = sorted(list(glob(config["new_data_path"]+"/video*")))
        for folder in folders:
            ref = sorted(glob(folder+"/rgb/*.png"))[0]
            ags.get_gscoords_for_video(folder, ref)
    else: 
        print("Please enter either 'yes' or 'no' as 'default_reference' field in json")

    # visualize the old and new coordinates
    old_path = config["old_data_path"]
    old_color = [0,0,255] #red

    new_path = config["new_data_path"]
    new_color = [0,255,255] #yellow

    print_coords(old_path, new_path , old_color, new_color)



class AlignGreenScreen():
    def __init__(self, **kwargs):
        print(kwargs)
        # print(kwargs)
        #PARAMETERS
        self.orb_max_matches = kwargs["orb_max_matches"]  # 1e7
        self.orb_max_iters = kwargs["orb_max_iters"]

        self.orb_scale_factor = kwargs["orb_scale_factor"]
        self.orb_levels = kwargs["orb_levels"]
        gaussian_k = kwargs["orb_gaussian_ksize"]
        self.orb_gaussian_ksize = [ gaussian_k, gaussian_k]
        self.orb_good_match_percent = kwargs["orb_good_match_percent"]

        self.ORB_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING) 
        ######
        self.ORB = cv2.ORB_create(
            nfeatures=int(self.orb_max_matches),
            nlevels=self.orb_levels,
            scaleFactor=self.orb_scale_factor,
        )
        
    def _preprocess_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame before keypoint detection
        1. Blur frames
        2. Convert to grayscale
        """
        # blur frames
        if self.orb_gaussian_ksize is not None:
            img = cv2.GaussianBlur(img, self.orb_gaussian_ksize, cv2.BORDER_DEFAULT)
            # img = cv2.GaussianBlur(img)

        # convert frames to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
        
    def _detect_keypoints(self, img: np.ndarray):
        """
        Detect keypoints of the given frame, and return the keypoint coordinates and descriptors
        img: (H, W, C)
        """
        # preprocess frame before keypoint detection
        img_pre = self._preprocess_frame(img)

        # orb keypoint detection
        kpts, descs = self.ORB.detectAndCompute(img_pre, None)
        return kpts, descs
    
    def _get_homography(self, kpts1: np.ndarray, descs1: np.ndarray, kpts2: np.ndarray, descs2: np.ndarray) -> np.ndarray:
        """
        Find homography between 2 frames
        kpts: detected keypoints
        descs: computed descriptors
        """
        # match features and sort by score
        matches = self.ORB_matcher.match(descs1, descs2, None)
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # preserve good matches
        numGoodMatches = int(len(matches) * self.orb_good_match_percent)
        matches = matches[:numGoodMatches]

        # extract location of good matches
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)

        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # find homography matrix H
        H, _ = cv2.findHomography(pts1, pts2, cv2.USAC_ACCURATE, maxIters=int(self.orb_max_iters))
        assert H.shape == (3, 3)

        return H
    
    def _get_homography_list(self, frames: torch.Tensor, ref_idx: int) :
        """Warp frames with respect to the reference frame."""
        if frames.shape[-1] != 3:
            frames = torch.permute(frames, (0, 2, 3, 1))
        frames = torch.unsqueeze(frames, dim=0)

        K, N, H, W, _ = frames.shape
        # all frames should in range [0, 1]
        assert frames.max() <= 1.0, frames.max()
        assert K == 1  # batch size 1

        frames_np = frames.detach().cpu().squeeze().numpy()
        frames_np = (frames_np * 255.0).astype(np.uint8)

        # detect keypoint in all frames
        orb_list = list()
        for idx in range(N):
            kpts, descs = self._detect_keypoints(frames_np[idx])
            orb_list.append((kpts, descs))

        # compute and store all homography matrices
        homo_list = list()  # store all homography matrices
        for idx in range(N):
            # use reference frame as global coordinate
            h = self._get_homography(*orb_list[idx], *orb_list[ref_idx])
            homo_list.append(h)
        homo_list = torch.from_numpy(np.array(homo_list)).to(frames.device)

        return homo_list
    

    def get_gscoords_for_video(self, path ,ref):
        """
        Given 
            path to video folder 
            the name of the reference image (image with correct green screen coordinates)
        This function rewrites all the coordinate files in the given 
        video path to the coordinates calculated using homography
        """
        all_imgs = sorted(glob(path+"rgb/*"))
        ref_img = cv2.imread(ref)
        ref_num = ref.split("/")[-1][:-4]
        ref_coords = np.load(path+"/patch_metadata/"+ ref_num+ ".npy")
        
        kp1 , desc1 = self._detect_keypoints(img=ref_img)

        for img in all_imgs:
            curr_img = cv2.imread(img)
            transformed_curr_gscoords = np.round(self.gscoords_frame(kp1, desc1 , ref_coords , curr_img))
            s = transformed_curr_gscoords.shape
            transformed_curr_gscoords = np.reshape(transformed_curr_gscoords,(s[0],s[2]) )

            #get img name
            img_name = img.split("/")[-1].split(".")[0]
            np.save(path+"patch_metadata/"+img_name+".npy"  ,transformed_curr_gscoords )
        return 
        
    
    def gscoords_frame(self, kp1 , desc1 , gscoords1 , img2 ):
        """
        Given: 
            key points, desc , and green screen coordinates for reference
            image 
        This function returns the projected reference coordinates in the given image
        """
        kp2 , desc2 = self._detect_keypoints(img=img2)
        H = self._get_homography(kpts1=kp1,descs1=desc1 , kpts2=kp2 , descs2=desc2)
        gscoords1 = np.float32(gscoords1).reshape(-1,1,2)
        transform_gscoords2 = cv2.perspectiveTransform(gscoords1 , H)
        return transform_gscoords2


    
def print_coords(path_old, path_new, clr_old , clr_new):
        """
        Given the path to older and newer dataset folders 
        and colours with which to display older and newer coordinates  
        This function creates a new folder in the newer dataset 'rgb_w_coords'
        that visualizes the green screen coordinates in both the old and new dataset 
        """
        all_imgs = sorted(list(glob(path_new+"/*/rgb/*.png")))
        all_new_coords = sorted(list(glob(path_new+"/*/patch_metadata/*.npy")))
        all_old_coords = sorted(list(glob(path_old+"/*/patch_metadata/*.npy")))
        # print(all_imgs[0] , all_imgs[1] )
        assert len(all_imgs) == len(all_old_coords)

        for i in range(len(all_imgs)):
            img = cv2.imread(all_imgs[i])
            # print(img.shape)
            old_coords = np.load(all_old_coords[i])
            for c in old_coords:
                # print(coords ,c)
                # print(c)
                img[int(c[1]) ,int(c[0])] = clr_old

            new_coords = np.load(all_new_coords[i])
            for c in new_coords:
                # print(coords ,c)
                # print(c)
                img[int(c[1]) ,int(c[0])] = clr_new
            
            p_to_img = all_imgs[i].split("/rgb/")[0] +"/rgb_w_coords"
            img_name = all_imgs[i].split("/rgb/")[1]
            # print(img_name)
            try:
                os.mkdir(p_to_img)
            except:
                 pass
            p = p_to_img + "/" + img_name 
            print(p)
            cv2.imwrite(p, img)
  

if __name__ == "__main__":
    sys.exit(_run(sys.argv))