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
rec_data=pd.read_csv("games_dataset/recommendations.csv", index_col=0)
rec_data.insert

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=65e2a800-7967-42b4-96c9-bf5538064c40' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>