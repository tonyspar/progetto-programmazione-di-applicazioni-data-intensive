# %% [markdown]
# # da commentare

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
    "matplotlib",
    "scikit-surprise",  # This is the correct package name for 'surprise'
    "jovian --upgrade --quiet",
    "kaggle",
    "numpy.typing"
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
import matplotlib.pyplot as plt
#import surprise
import jovian
from sklearn.datasets import fetch_openml
# %%
#NON RIESEGUIRE
# if not os.path.exists("kaggle.json") and not os.path.exists("~/.kaggle/kaggle.json"):
#     ! mkdir ~/.kaggle
#     ! cp kaggle.json ~/.kaggle/
#     ! chmod 600 ~/.kaggle/kaggle.json
#     ! kaggle datasets list
#     !kaggle datasets download antonkozyriev/game-recommendations-on-steam
#     !mkdir games
#     !unzip game-recommendations-on-steam.zip -d games_dataset

# %% [markdown]
# ## dataset principale

# %%
bigdata=pd.read_csv("games_dataset/games.csv", index_col=0)
bigdata['steam_deck'].value_counts()
bigdata.drop(columns=["steam_deck"], inplace=True);
bigdata

# %% [markdown]
# ## editing dataset dati aggiuntivi

# %% [markdown]
# Alcune feature non sono rilevanti per il nostro problema, possiamo quindi rimuovere le colonne dal dataframe per risparmiare ulteriormente spazio.

# %%
# editing dataset dati aggiuntivi
df = fetch_openml(data_id=43689)
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
tagsdata.index

# %%
# merge dei dataset
tempdata=pd.merge(how='left', left=bigdata, right=secdata, left_index=True, right_index=True)
data=pd.merge(how='left', left=tempdata, right=tagsdata, left_index=True, right_index=True)
data.drop(columns='Tags', inplace=True)
data
# %%

def rank_stats(value):
    if value == 'true':
        return random.randint(3, 5)
    else:
        return random.randint(1, 2)

rec_data=pd.read_csv("games_dataset/recommendations.csv", index_col=0)
rec_data.insert(0, 'Rank', rec_data.index)
rec_data['Rank'] = rec_data['is_recommended'].map(rank_stats)
rec_data

# %%
