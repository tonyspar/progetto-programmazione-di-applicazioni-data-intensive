# %% [markdown]
# # da commentare

# %%
import sys
import subprocess

# %%
def run_cmd(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        print(f"Command failed: {cmd}\n{e}")

def install(package):
    # Try pip first
    try:
        run_cmd(f"{sys.executable} -m pip install --upgrade pip")
        run_cmd(f"{sys.executable} -m pip install {package}")
    except Exception:
        # Fallback for environments like Colab
        run_cmd(f"!pip install {package}")

# List of required packages
packages = [
    "numpy<2.0,>=1.20",
    "pandas",
    "scikit-learn",
    "matplotlib", # This is the correct package name for 'surprise'
    "jovian --upgrade --quiet",
    "kaggle",
    "numpy.typing",
    "os"
]

def is_conda():
    try:
        return hasattr(sys, 'base_prefix') and 'conda' in sys.version or 'Continuum' in sys.version
    except Exception:
        return False

def install(package):
    if is_conda():
        # Try installing with conda first
        try:
            run_cmd(f"conda install -y {package.split()[0]}")
            return
        except Exception:
            pass
    # Fallback to pip
    try:
        run_cmd(f"{sys.executable} -m pip install --upgrade pip")
        run_cmd(f"{sys.executable} -m pip install {package}")
    except Exception:
        # Fallback for environments like Colab
        run_cmd(f"!pip install {package}")

# Uninstall numpy first to avoid version conflicts
#if not os.path.exists(r"C:\Users\manto\\"):
    run_cmd(f"{sys.executable} -m pip uninstall numpy -y")

    for pkg in packages:
        install(pkg)

# %% [markdown]
# ## 1. Descrizione del problema e analisi esplorativa
# ### Caricamento Librerie
# Per prima cosa carichiamo le librerie per effettuare operazioni sui dati
# 

# %% [markdown]
# NumPy per creare e operare su array a N dimensioni
#     pandas per caricare e manipolare dati tabulari
#     matplotlib per creare grafici
# 
# Importiamo le librerie usando i loro alias convenzionali e abilitando l'inserimento dei grafici direttamente nel notebook
#


# %%
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#%matplotlib inline
#import surprise
import jovian
from sklearn.datasets import fetch_openml
import os
# %%
#NON RIESEGUIRE
# if not os.path.exists("kaggle.json") and not os.path.exists("~/.kaggle/kaggle.json"):
     #! mkdir ~/.kaggle
     #! cp kaggle.json ~/.kaggle/
     #! chmod 600 ~/.kaggle/kaggle.json
     #! kaggle datasets list 
     #! kaggle datasets download antonkozyriev/game-recommendations-on-steam 
     #! unzip game-recommendations-on-steam.zip -d games_datasetco

# %% [markdown]
# ## Data cleaning
# %%
# if os.path.exists(r"C:\Users\manto\\"):
#     print(os.getcwd())
#     os.chdir(r"C:\Users\manto\Documents\Università\3°anno\Data_intensive\progetto-programmazione-di-applicazioni-data-intensive")
#     print(os.getcwd())
# %%
bigdata=pd.read_csv("./games_dataset/games.csv", index_col=0)
#bigdata['steam_deck'].value_counts()
bigdata.drop(columns=["steam_deck"], inplace=True)
def rank_stats_stream(value):
    mapping = {
        'overwhelmingly positive': 9-5,
        'very positive': 8-5,
        'mostly positive': 7-5,
        'positive': 6-5,
        'mixed': 5-5,
        'negative': 4-5,
        'mostly negative': 3-5,
        'overwhelmingly negative': 1-5,
        'very negative': 2-5
    }
    return mapping.get(str(value).lower(), np.nan)

bigdata["rating"]= bigdata["rating"].map(rank_stats_stream)
bigdata['rating'].unique()
#rec_data['is_recommended'] = rec_data['is_recommended'].map(rank_stats)

# %% [markdown]
# ## editing dataset dati aggiuntivi

# %% [markdown]
# Alcune feature non sono rilevanti per il nostro problema, possiamo quindi rimuovere le colonne dal dataframe per risparmiare ulteriormente spazio.

# %%
# editing dataset dati aggiuntivi
df = fetch_openml(data_id=43689)
#Genres tolti perché saranno già inclusi in tag
secdata = pd.DataFrame({
    "SteamURL": df.data["SteamURL"],
    "Metacritic": df.data["Metacritic"],
    "Platform": df.data["Platform"],
    "Tags": df.data["Tags"],
    "Languages": df.data["Languages"]
})
secdata=secdata.dropna(axis='rows', subset=["SteamURL","Tags"])
secdata['SteamURL']=secdata['SteamURL'].str[35:-16]
secdata.drop_duplicates(subset=['SteamURL'], inplace=True)
secdata['SteamURL']=secdata['SteamURL'].astype('int64')
secdata.index=secdata['SteamURL']
secdata.drop(columns='SteamURL', inplace=True)

# %%
# editing dataset tag
metadata = pd.read_json("games_dataset/games_metadata.json", lines="True")
metadata.to_csv("data3.csv", encoding='utf-8', index=False)
tagsdata = pd.read_csv("data3.csv", index_col=0)
tagsdata=tagsdata.drop(columns='description')
tagsdata.drop(tagsdata[tagsdata['tags'].str.len()==2].index, inplace=True)
tagsdata


# %%
rec_data = pd.read_csv("games_dataset/recommendations.csv", index_col=0)

# %% [markdown]

random.seed(42)

def rank_stats(value):
    if str(value).lower() == 'true':
        return random.randint(3, 5)
    else:
        return random.randint(1, 2)

def reduce3(value):
    if value == 3:
        return random.randint(3,5)
    else:
        return value

rec_data.columns
rec_data.drop(columns=['date','funny','review_id'], inplace=True)
#rec_data['is_recommended'] = rec_data['is_recommended'].map(rank_stats)
#rec_data=rec_data.rename(columns={'is_recommended': 'rank'})
#rec_data['rank'] = rec_data['rank'].map(reduce3)
#rec_data['rank'].value_counts()

# %% [markdown]
rec_data['is_recommended'] = rec_data['is_recommended'].map(rank_stats)
rec_data=rec_data.rename(columns={'is_recommended': 'rank'})
rec_data['rank'] = rec_data['rank'].map(reduce3)
#rec_data['rank'].value_counts()



# %%
# merge dei dataset
tempdata=pd.merge(how='left', left=bigdata, right=secdata, left_index=True, right_index=True)
data=pd.merge(how='left', left=tempdata, right=tagsdata, left_index=True, right_index=True)
#data=pd.merge(how='left', left=data, right=rec_data, left_index=True, right_index=True)
data.drop(columns='Tags', inplace=True)

# %% [markdown]
#Rilevazione di valori nulli
#matrice.notna()
#Nello specifico, le features disponibile (come si può osservare dalla rappresentazione del dataset) sono:

    # Age | Objective Feature | age | int (days)
    # Height | Objective Feature | height | int (cm) |
    # Weight | Objective Feature | weight | float (kg) |
    # Gender | Objective Feature | gender | categorical code |
    # Systolic blood pressure | Examination Feature | ap_hi | int |
    # Diastolic blood pressure | Examination Feature | ap_lo | int |
    # Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
    # Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
    # Smoking | Subjective Feature | smoke | binary |
    # Alcohol intake | Subjective Feature | alco | binary |
    # Physical activity | Subjective Feature | active | binary |
    # Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

#dataset['cardio'].value_counts().plot.pie(autopct='%1.1f%%')

# %%
#grafico dei generi che hanno ricevuto piu' voti (torta e barre)
#grafico giochi piu' votati(data.head().pie)blabla
#media dei voti per genere

# %%
xGrafico=rec_data.join(tagsdata, on='app_id', how='left')
# %%
# Ogni riga di 'tags' è una stringa di tag separati da virgole, quindi bisogna splittare, unire e trovare i valori unici
xGrafico=xGrafico.dropna(subset=['tags'])
#num_unique_tags = all_tags.nunique()
#print(num_unique_tags)

# %%
# Numero di giochi con numero di recensioni dentro un certo range
xGrafico.index.value_counts()
#pd.qcut(xGrafico.index.value_counts(), 10).value_counts().sort_index().plot.bar()
#pd.qcut(xGrafico.index.value_counts(), 10).value_counts().sort_index().plot.pie()
xGrafico.index.value_counts().plot.box(figsize=(4, 6))



# free_color = "#2ca02c" # green
# occupied_color = "#d32829" # red

# class_colors = [free_color, occupied_color]
# class_colors_map = {0:free_color, 1:occupied_color}

# print(data["Occupancy"].value_counts())
# _ = data["Occupancy"].value_counts().plot.pie(colors=class_colors)
# %%
def plot_bar(feature, n, title):
    wine[feature].value_counts()[:n].plot.bar(figsize=(15, 4))
    plt.axes().set_title(title)
    plt.show()

plot_bar("country", 20, "Paesi più frequenti")


# %%
rec_data_cut=xGrafico.drop(columns=["tags"])
rec_data_cut
# %%
rec_data_cut.to_csv("games_dataset/rec_cut.csv")