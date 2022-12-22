#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import random
import shutil

base_dir = "Images"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
        
        
def split(base_dir, train_percent, val_percent, test_percent):
    # Set the directories
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    # Set the percentages for the splits
#     train_percent = 0.6
#     val_percent = 0.2
#     test_percent = 0.2

    # Create the train, val, and test directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Loop through the class directories
    for class_dir in os.listdir(base_dir):
        if class_dir in ["train", "val", "test"]:
            continue


            # Check if the item is a directory
        if not os.path.isdir(os.path.join(base_dir, class_dir)):
            # Skip the item if it is not a directory
            continue

        # Get the list of files in the class directory
        class_files = os.listdir(os.path.join(base_dir, class_dir))

        # Shuffle the files
        random.shuffle(class_files)

        # Calculate the number of files for each split
        num_train = int(len(class_files) * train_percent)
        num_val = int(len(class_files) * val_percent)
        num_test = len(class_files) - num_train - num_val

        # Split the files into the different splits
        train_files = class_files[:num_train]
        val_files = class_files[num_train:num_train+num_val]
        test_files = class_files[num_train+num_val:]

        # Create the class directories in the train and val directories
        train_class_dir = os.path.join(train_dir, class_dir)
        val_class_dir = os.path.join(val_dir, class_dir)
        test_class_dir = os.path.join(test_dir, class_dir)
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        # Copy the files to the appropriate directories
        for file in train_files:
            shutil.copy(os.path.join(base_dir, class_dir, file), train_class_dir)
        for file in val_files:
            shutil.copy(os.path.join(base_dir, class_dir, file), val_class_dir)
        for file in test_files:
            shutil.copy(os.path.join(base_dir, class_dir, file), test_class_dir)


def delete_original_folders(base_dir, class_dirs):
    for class_dir in class_dirs:
        dir_path = os.path.join(base_dir, class_dir)
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                os.unlink(os.path.join(root, name))
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    # Directory is not empty
                    pass
        try:
            os.rmdir(dir_path)
        except OSError:
            # Directory is not empty
            pass

# Set the base directory and the list of class directories to delete

class_dirs = ["Pebbles", "Shells"]


split(base_dir, 0.6, 0.2, 0.2)
# Delete the original folders
delete_original_folders(base_dir, class_dirs)


# In[ ]:




