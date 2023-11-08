import pandas as pd
import data_preprocess
import prediction
from glob import glob
label_data = pd.read_csv('./label.csv', encoding='cp949')
file_list = glob('./data/*.mp4')
data_preprocess.Preprocessing(file_list, label_data)
image_list = glob('../../data/1-cycle_30%_중간데이터/whole_tile/image/*.tiff')
total_path, total_prob, total_dice = prediction.predict(image_list)
df = pd.DataFrame(columns=['ID', 'true_label', 'predict'])
classes = ['Oropharynx', 'Tonguebase', 'Epiglottis']
for i in range(len(path_list)):
    df.loc[i] = [path_list[i][0], classes[total_y[i].item()],
                 classes[total_prob[i].item()]]
df.to_csv('./data/predict.csv')
df_cm = pd.DataFrame(cm, columns=['Oropharynx', 'Tonguebase', 'Epiglottis'], index=[
                     'Oropharynx', 'Tonguebase', 'Epiglottis'])
df_cm.to_csv('./data/cm.csv')
