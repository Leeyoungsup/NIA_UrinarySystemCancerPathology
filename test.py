import pandas as pd
import data_preprocess
import prediction
from glob import glob
label_data = pd.read_csv('./label.csv', encoding='cp949')
file_list = glob('./data/*.mp4')
data_preprocess.Preprocessing(file_list, label_data)
path_list, total_y, total_prob, cm = prediction.predict(label_data)
df = pd.DataFrame(columns=['ID', 'true_label', 'predict'])
classes = ['Oropharynx', 'Tonguebase', 'Epiglottis']
for i in range(len(path_list)):
    df.loc[i] = [path_list[i][0], classes[total_y[i].item()],
                 classes[total_prob[i].item()]]
df.to_csv('./data/predict.csv')
df_cm = pd.DataFrame(cm, columns=['Oropharynx', 'Tonguebase', 'Epiglottis'], index=[
                     'Oropharynx', 'Tonguebase', 'Epiglottis'])
df_cm.to_csv('./data/cm.csv')
