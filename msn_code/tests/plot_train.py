
#%%
import yaml
import matplotlib as plt
import pandas as pd
import os


# fname= 
# with open(fname, 'r') as y_file:
#     params = yaml.load(y_file, Loader=yaml.FullLoader)


# file_path = os.path.join(params['logging']['folder'],f"{params['logging']['write_tag']}_r0.csv")

file_path = "/home/livia/Documentos/Projetos/climate-clustering/msn_code/checkpoint/msn-temperature-3p/msn-temperature-3p_r0.csv"

data= pd.read_csv(file_path)
data.set_index("epoch")['msn'].plot()

