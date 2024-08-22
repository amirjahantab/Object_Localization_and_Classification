import os
import glob
import shutil
import random
from tqdm import tqdm # generation of progress bars


def create_validation_data(trn_dir, val_dir, split=0.1, ext='png'):
    '''
    Moves a subset of images from the training directory to the validation directory based on a specified split ratio.
    
    Parameters:
    -----------
    trn_dir : str
        The directory containing the training data. This directory should have subdirectories for each class.
    
    val_dir : str
        The directory where the validation data will be stored. Subdirectories for each class will be created here if they do not already exist.
    
    split : float or int, optional
        The fraction (if float) or absolute number (if int) of images to be moved to the validation directory. 
        If a float is provided, it should be between 0 and 1, representing the proportion of images to move.
        If an integer is provided, it represents the exact number of images to move.
        The default value is 0.1 (that is 10% of the images will be moved).
    
    ext : str, optional
        The file extension of the images to be processed (e.g., 'png', 'jpg'). The default is 'png'.
    '''
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
        
    train_ds = glob.glob(trn_dir + f'/*/*.{ext}') # Generate list of image files
    print(len(train_ds))
    
    valid_sz = int(split * len(train_ds)) if split < 1.0 else split 
    
    valid_ds = random.sample(train_ds, valid_sz) # This function selects valid_sz images randomly from train_ds without replacement.
    print(len(valid_ds))
    
    for fname in tqdm(valid_ds):
        basename = os.path.basename(fname) # Extracts the base name (file name with extension) from the full file path fname.
        label = fname.split('\\')[-2] # Extracts the label 
        src_folder = os.path.join(trn_dir, label)
        tgt_folder = os.path.join(val_dir, label)
        if not os.path.exists(tgt_folder):
            os.mkdir(tgt_folder)
        shutil.move(os.path.join(src_folder, basename), os.path.join(tgt_folder, basename))