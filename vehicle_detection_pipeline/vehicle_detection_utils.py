from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage.measurements import label

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True, hog_channel=3):
    # return with two outputs vis==True
    features = []
    hog_images = []
    if vis == True:
        for channel in range(hog_channel):
            feature, hog_image = hog(img[:,:,channel], orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                block_norm='L2-Hys',
                                transform_sqrt=False,
                                visualise=True,
                                feature_vector=False)
            features.append(feature)
            hog_images.append(hog_image)
        return features, hog_images
    # Otherwise call with one output
    else:
        for channel in range(hog_channel):
            feature = hog(img[:,:,channel], orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                block_norm='L2-Hys',
                                transform_sqrt=False,
                                visualise=False,
                                feature_vector=False)
            features.append(feature)
        return features

def convert_color(img, conv='BGR2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def extract_hog_features_from_img_list(imgs, cspace='RGB', cell_per_block=2, pix_per_cell=8, oritentation_bins=12, hog_channel=3):
    """
        Extract hog features from a image
    """
    assert(imgs is not None)
    features_arr = []
    for img_file in imgs:
        img = mpimg.imread(img_file)
        features = []
        for channel in range(hog_channel):
            feature = hog(img[:,:,channel], orientations=oritentation_bins,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                block_norm='L2-Hys',
                                transform_sqrt=False,
                                visualise=False,
                                feature_vector=False)
            features.append(feature)
        features = np.ravel(features)
        features_arr.append(features)
    return features_arr


def normalize_feature_vector(feature_arr):
    """
        Normalize items in feature vector
        feature_arr: should be a list where each row is a feature vector (that is unravel from hog features)
    """
    assert(feature_arr is not None)
    scaler = StandardScaler().fit(feature_arr)
    scaled_x = scaler.transform(feature_arr)
    return scaled_x

def color_hist(img, num_bins=32, bins_range=(0, 256)):
    """
        Compute histogram of RGB channel seperately
        :return concatenate R,G,B histograms into single feature vector
    """
    assert(img.shape[2] == 3) # assert RGB channel
    r_hist = np.histogram(img[:,:,0], bins=num_bins, range=bins_range)
    g_hist = np.histogram(img[:,:,1], bins=num_bins, range=bins_range)
    b_hist = np.histogram(img[:,:,2], bins=num_bins, range=bins_range)
    rgb_feature = np.concatenate((r_hist[0], g_hist[0], b_hist[0]))
    return rgb_feature

def bin_spatial(img, size=(32,32)):
    """
        Rresize and ravel images into 1d array as features
    """
    # resize image and unpack features onto 1d array
    features = cv2.resize(img, size).ravel()
    return features

def extract_featurres_from_img_list(img_list, cspace='BGR2YCrCb',
                                    spatial_size=(32,32),
                                    hist_bins=32,
                                    hist_range=(0,256),
                                    orient=9,
                                    pix_per_cell=8,
                                    cell_per_block=2
                                   ):
    feature_list = []
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = convert_color(img, conv=cspace)
        img = img.astype(np.float32)/255.0
        bin_spatial_feature = bin_spatial(img, spatial_size)
        color_hist_feature = color_hist(img, num_bins=hist_bins, bins_range=(0, 256))
        hog_features = get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True, hog_channel=3)
        hog_features = np.hstack((hog_features[0].ravel(), hog_features[1].ravel(), hog_features[2].ravel()))
        #print(hog_features.shape)
        full_features = np.hstack((bin_spatial_feature, color_hist_feature, hog_features)).reshape(1, -1)
        #feature_list.append(np.concatenate((bin_spatial_feature, color_hist_feature)))
        feature_list.append(full_features)
    return feature_list

def get_scaler_from_feature_vectors (feature_vectors):
    #X = np.vstack(feature_vectors).astype(np.float64)
    X_scaler = StandardScaler().fit(feature_vectors)
    return X_scaler


def slide_windows_and_update_heat_map(img, ystart,
                                      ystop, scale, svc,
                                      X_scaler, orient,
                                          pix_per_cell,
                                          cell_per_block,
                                          spatial_size,
                                          hist_bins,
                                          heat_map,
                                          threshold=3,
                                          window_size=64,
                                          cells_per_step=2,
                                          ):
    # using cv2 to read img in so normalize it here
    img = np.copy(img)
    img = img.astype(np.float32)/255.0

    # select interest area
    img_to_search = img[ystart:ystop,:,:]
    if scale > 1:
        imshape = img_to_search.shape
        img_to_search = cv2.resize(img_to_search, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # convert to correct color space
    img_to_search = convert_color(img_to_search, conv='BGR2YCrCb')
    # caclculate number of blocks and steps
    nxblocks = (img_to_search.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img_to_search.shape[0] // pix_per_cell) - cell_per_block + 1

    cells_per_window = 64 // pix_per_cell
    nblocks_per_window = cells_per_window - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # extract all hog features from channel
    hog_features = get_hog_features(img_to_search,
                                        cell_per_block=cell_per_block,
                                        pix_per_cell=pix_per_cell,
                                        orient=orient,
                                        feature_vec=False,
                                        hog_channel=3)
    print("hog shape", hog_features[0].shape)
    # hog_feature_shapes = (nxblocks, nyblocks, cell_per_block, cell_per_block, 3)
    # sliding windows here
    bboxes = []
    window_count = 0
    for xidx in range(nxsteps):
        for yidx in range(nysteps):
            window_count += 1
            ypos = yidx * cells_per_step
            xpos = xidx * cells_per_step
            # extract feature from each channel
            feat1 = hog_features[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
            feat2 = hog_features[1][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
            feat3 = hog_features[2][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
            #print(feat1.shape, feat2.shape, feat3.shape)
            hog_feats = np.hstack((feat1.ravel(), feat2.ravel(), feat3.ravel()))
            #print("hog shapes:",hog_feats.shape)
            # extract color features from the corresponding windows
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            sub_img = cv2.resize(img_to_search[ytop:ytop+window_size, xleft:xleft+window_size],(64, 64))
            spatial_features = bin_spatial(sub_img, size=spatial_size)
            hist_features = color_hist(sub_img, num_bins=hist_bins, bins_range=(0, 256))
            # combine all features
            #print (spatial_features.shape, hist_features.shape, hog_feats.shape)
            features_to_scale = np.hstack((spatial_features, hist_features, hog_feats)).reshape(1, -1)
            #print("features to scale", features_to_scale.shape)
            all_features = X_scaler.transform(features_to_scale)
            prediction = svc.predict(all_features)
            decision = svc.decision_function(all_features)
            if (prediction == 1) and (decision > 0.5):
                print("Decision: %.2f" % decision)
                window_draw = np.int(window_size * scale)
                x_draw = np.int(xleft * scale)
                y_draw = np.int(ytop * scale)
                x1 = x_draw
                y1 = y_draw+ystart
                x2 = x_draw+window_draw
                y2 = y_draw+window_draw+ystart
                bboxes.append(((x1, y1),(x2, y2)))
                cv2.rectangle(img, (x1, y1),(x2, y2), (0,255,0), 10)
                #print(x1,x2,y1,y2)
                heat_map[y1:y2, x1:x2] += 1
    # apply threshold for heat map here
    #heat_map[heat_map <= threshold] = 0
    #labels = label(heat_map)
    print ("Total Window count %d" % window_count)
    return heat_map, img

def apply_heat_map_threshold(heat_map, threshold):
    assert(heat_map is not None)
    heat_map[heat_map <= threshold] = 0

def draw_bounding_boxes_from_labels(img, labels, area_threshold=2000):
    # draw bounding boxes based on label
    for car_idx in range(1, labels[1] + 1):
        nonzero_idx = (labels[0] == car_idx).nonzero()
        nonzeroy = np.array(nonzero_idx[0])
        nonzerox = np.array(nonzero_idx[1])
        bbox = ((np.min(nonzerox),np.min(nonzeroy)),( np.max(nonzerox), np.max(nonzeroy)))
        bbox_area = np.abs((bbox[0][0]-bbox[1][0]) * (bbox[0][1]-bbox[1][1]))
        print ("BBox : ",bbox," bbox area:",bbox_area)
        if bbox_area > area_threshold:
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 10)
    return img

