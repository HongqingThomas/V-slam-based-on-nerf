import rosbag
import rospy
from cv_bridge import CvBridge
import numpy as np
import cv2 
import os
from sensor_msgs.msg import Image
import shutil

ros_bag = "Rosbag/parking_lot_0430.bag"
topic1 = "/zed2/zed_node/depth/depth_registered"
topic2 = "/zed2/zed_node/left/image_rect_color"
topic3 = "/zed2/zed_node/right/image_rect_color"
topic4 = "/zed2/zed_node/odom"

root = "./Result/"
all_mode = 'all'
train_mode = 'training'
test_mode = 'testing'
sample_mode = "sample"
folder_path1 = root + all_mode + '/depth/'
folder_path2 = root + all_mode + '/left_rgb/'
folder_path3 = root + all_mode + '/right_rgb/'
folder_path4 = root + all_mode + '/odom/'

folder_list = [folder_path1, folder_path2, folder_path3, folder_path4]

bridge = CvBridge()
bag = rosbag.Bag(ros_bag)

# Write RGB and Depth image
def cv_imwrite(img_path, msg, time_step, type):
    if type == "rgb":
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_name = str(time_step) + '.jpg'
        cv2.imwrite(img_path + image_name, cv_image)
    elif type == "depth":
        cv_image = bridge.imgmsg_to_cv2(msg, "32FC1")
        # cv_image = np.where(np.isnan(cv_image), 0, cv_image)
        # cv_image = np.where(np.isinf(cv_image), 0, cv_image)
        # # uint8 will normalize data into 256 values, 
        # if the value is larger than 256, its better to use uint16
        cv_image = (cv_image * 1000.0).astype(np.uint16)
        # To visualize better: Scale the values in the matrix to the range [0, 255]
        # scaled_matrix = (cv_image / np.max(cv_image)) * 255
        image_name = str(time_step) + '.png'
        cv2.imwrite(img_path + image_name, cv_image)
    elif type == "txt":
        image_name = str(time_step) + '.txt'
        with open(img_path + image_name, "w+") as file:
            file.write(str(msg))


# Transfer data(image) from ros to cv2
def generateAllFrame():
    topic1_messages = []
    topic2_messages = []
    topic3_messages = []
    topic4_messages = []
    # Create folders
    for path_name in folder_list:
        if not os.path.exists(path_name):
            os.makedirs(path_name)
    # Write Images
    num = 0
    for topic, msg, t in bag.read_messages(topics=[topic1, topic2, topic3, topic4]):
        if topic == topic1:
            topic1_messages.append(msg)
        elif topic == topic2:
            topic2_messages.append(msg)
        elif topic == topic3:
            topic3_messages.append(msg)
        elif topic == topic4:
            topic4_messages.append(msg)
        # Synchronize messages from topic1 and topic2 based on their timestamps
        if len(topic1_messages) > 0 and len(topic2_messages) > 0 and len(topic3_messages) > 0 and len(topic4_messages) > 0:
            stamp1 = topic1_messages[-1].header.stamp
            stamp2 = topic2_messages[-1].header.stamp
            stamp3 = topic3_messages[-1].header.stamp
            stamp4 = topic4_messages[-1].header.stamp
            # stamp4 = topic4_messages[-1].header.stamp
            if stamp1 == stamp2 and stamp2 == stamp3 and stamp3 == stamp4:
                cv_imwrite(folder_list[0], topic1_messages[-1], num, "depth") # Depth
                cv_imwrite(folder_list[1], topic2_messages[-1], num, "rgb") # Left rgb
                cv_imwrite(folder_list[2], topic3_messages[-1], num, "rgb") # right rgb
                cv_imwrite(folder_list[3], topic4_messages[-1], num, "txt") # odom
                topic1_messages.pop()
                topic2_messages.pop()
                topic3_messages.pop()
                topic4_messages.pop()
                num += 1
                print("Number: ", num)
    bag.close()

# Copy data to new folder
def copyData(idx, folder_path, mode, file_type):
    path_list = folder_path.split(all_mode)
    sample_folder_path =  path_list[0] + mode + path_list[1]
    file_name = str(idx) + file_type
    src_path = os.path.join(folder_path, file_name)
    dst_path = os.path.join(sample_folder_path, file_name)
    shutil.copy(src_path, dst_path)

# create folder according to mode: root+mode+topic_name
def createFolder(folder_list, mode):
    # Create the sample folder if it doesn't exist
    for folder_name in folder_list:
        path_list = folder_name.split(all_mode)
        folder_name =  path_list[0] + mode + path_list[1]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)


# shuffle data into train and test 
def shuffleData(ratio, step):
    # Get a list of all image files in the folder
    image_files1 = [f for f in os.listdir(folder_list[0]) if f.endswith('.png') or f.endswith('.jpg')]
    image_files2 = [f for f in os.listdir(folder_list[1]) if f.endswith('.png') or f.endswith('.jpg')]
    image_files3 = [f for f in os.listdir(folder_list[2]) if f.endswith('.png') or f.endswith('.jpg')]
    if (len(image_files1) == len(image_files2) and len(image_files2) == len(image_files3)):
        length = len(image_files1)
        data = np.arange(length)
        np.random.shuffle(data)
        train_size = int(length * ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        print("train_data: ", len(train_data) / step)
        for train_idx in train_data:
            if train_idx % step == 0:
                mode = train_mode
                createFolder(folder_list, mode)
                copyData(train_idx, folder_list[0], mode, ".png")
                copyData(train_idx, folder_list[1], mode, ".jpg")
                copyData(train_idx, folder_list[2], mode, ".jpg")
        print("test_data: ", len(test_data) / step)
        for test_idx in test_data:
            if test_idx % step == 0:
                mode = test_mode
                createFolder(folder_list, mode)
                copyData(test_idx, folder_list[0], mode, ".png")
                copyData(test_idx, folder_list[1], mode, ".jpg")
                copyData(test_idx, folder_list[2], mode, ".jpg")
    else:
        print("Size not the same")

# Sample data every sample_size
def sampleData(sample_size):
    # Get a list of all image files in the folder
    image_files1 = [f for f in os.listdir(folder_list[0]) if f.endswith('.png') or f.endswith('.jpg')]
    image_files2 = [f for f in os.listdir(folder_list[1]) if f.endswith('.png') or f.endswith('.jpg')]
    image_files3 = [f for f in os.listdir(folder_list[3]) if f.endswith('.txt')]
    if (len(image_files1) == len(image_files2)) and (len(image_files2) == len(image_files3)):
        # Calculate the step size based on the desired sample size and the number of images
        step_size = max(len(image_files1) // sample_size, 1)
        # Sample the images at a constant interval
        sampled_images1 = [image_files1[i] for i in range(0, len(image_files1), step_size)]
        sampled_images2 = [image_files1[i] for i in range(0, len(image_files2), step_size)]
        sampled_images3 = [image_files1[i] for i in range(0, len(image_files3), step_size)]
        if (len(image_files1) == len(image_files2)):
            print(len(sampled_images1))
            print(sampled_images1)
            # Copy the sampled images to the sample folder
            for file_name in sampled_images1:
                mode = sample_mode
                idx = file_name.split('.')[0]
                createFolder(folder_list, mode)
                copyData(idx, folder_list[0], mode, ".png")
                copyData(idx, folder_list[1], mode, ".jpg")
                copyData(idx, folder_list[3], mode, ".txt")

if __name__ == "__main__":
    # Transfer data frm ROS to images
    # generateAllFrame()

    # Shuffle data into train and test randomly
    # ratio = 0.8
    # step = 2
    # shuffleData(ratio, 2)

    # Sample data every sample_size
    # sample_size = 500
    # sampleData(sample_size)
