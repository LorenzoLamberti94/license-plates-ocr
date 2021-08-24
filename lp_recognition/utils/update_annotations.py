import os
import pandas as pd
from os.path import join


def modify_annotations():
    ################################################################################
    # Open and modify annotations of a csv file
    ################################################################################
    OLD_ANNOT_FILE = '/home/lamberti/work/Dataset/CCPD_crops/annotation_ccpd_base_LPRNet.csv'
    PATH_TO_ADD = '/home/lamberti/work/Dataset/CCPD_crops/'

    names = ['image_path','lp']
    df = pd.read_csv(OLD_ANNOT_FILE,  sep = " ",names=names)
    print('\n', df, '\n')

    # Remove Column
    # del df['lp']
    # print('\n', df, '\n')

    # Iterate to modify full paths of the images
    for i, row in df.iterrows():
        # print(row['image_path'])
        # df.iloc[i,0] = join ('ccpd_base/' , os.path.basename(df.iloc[i,0]) ) #keep only last part of the path and add something before
        df.iloc[i,0] = join( PATH_TO_ADD , df.iloc[i,0] ) #just add something before
        if i % 1000 == 0 :
            print(i)
        # if i == 100: # DEBUG: early stop!
        #     break

    # Shuffle rows
    # df = df.sample(frac=1).reset_index(drop=True)

    # WRITE CSV FILE IN LPRNet FORMAT
    header = ['image_path','lp']
    df.to_csv('new_annot.csv', columns=header, sep = " ", index=False, header=False)

    print(df)
    print('\n end')


    ''' Sample on how we want the annotations
    /home/lamberti/work/Dataset/synthetic_chinese_license_plates_LPRNet/crops/184539.png <Shandong>VB9WR1
    '''


def mix_two_datasets():
    ################################################################################
    # Append two csv files: take annotations for 2 datasets, unify the two datasets and shuffle the order
    ################################################################################

    FILE1 = '/home/lamberti/work/Dataset/CCPD_crops/train_ccpd_crops'
    FILE2 = '/home/lamberti/work/Dataset/synthetic_chinese_license_plates_LPRNet/train'
    names = ['image_path','lp']
    df1 = pd.read_csv(FILE1,  sep = " ",names=names)
    print('\n', df1, '\n')
    df2 = pd.read_csv(FILE2,  sep = " ",names=names)
    print('\n', df2, '\n')
    df_mixed = df1.append(df2)
    print('\n', df3, '\n')

    # Shuffle rows
    df_mixed = df_mixed.sample(frac=1).reset_index(drop=True)
    print('\n', df_mixed, '\n')

    # Save
    header = ['image_path','lp']
    df_mixed.to_csv('new_annot.csv', columns=header, sep = " ", index=False, header=False)

    print('\n', df_mixed, '\n')



if __name__ == '__main__':

    # modify_annotations()
    mix_two_datasets()
