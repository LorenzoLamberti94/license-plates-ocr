import pandas as pd
DATASET_DIR = '/home/lamberti/work/Dataset/Cars_czech/CarsReId_LP-dataset/'

READ_FILE_PATH ='/home/lamberti/work/Dataset/Cars_czech/CarsReId_LP-dataset/trainVal.csv'
# READ_FILE_PATH = '/home/lamberti/work/annotations.csv'

df = pd.read_csv(READ_FILE_PATH)
print('\n')

print(df)
print('\n')

print(df.dtypes)
print('\n')

# Iterate to give full paths to the images
for i, row in df.iterrows():
    # print(row['image_path'])
    df.iloc[i,1] = DATASET_DIR + df.iloc[i,1]
    if i % 1000 == 0 :
        print(i)
    # row['image_path'] = DATASET_DIR + row['image_path']
    # print(row['image_path'])
    if i == 100:
        break


# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# WRITE CSV FILE IN LPRNet FORMAT
header = ['image_path','lp']
df.to_csv('lprnet_annot.csv', columns=header, sep = " ", index=False, header=False)

print(df)
print('\n')

print('end')

# print(df.iloc[2, 2])


''' Sample on how we want the annotations
/home/lamberti/work/Dataset/synthetic_chinese_license_plates_LPRNet/crops/184539.png <Shandong>VB9WR1
'''
