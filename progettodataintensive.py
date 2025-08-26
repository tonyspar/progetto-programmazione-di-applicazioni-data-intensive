# %%
if(False):
    import sys
    import subprocess
    import platform

    def run_cmd(cmd):
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception as e:
            print(f"Command failed: {cmd}\n{e}")

    def is_windows():
        return platform.system() == "Windows"

    def install(package):
        if is_windows():
            # Su Windows, prova a usare conda
            try:
                run_cmd(f"conda install -y {package.split()[0]}")
                return
            except Exception:
                print(f"Conda installation failed for {package}, trying pip...")
        # Fallback a pip per tutti gli altri casi
        try:
            run_cmd(f"{sys.executable} -m pip install --upgrade pip")
            run_cmd(f"{sys.executable} -m pip install {package}")
        except Exception as e:
            print(f"Pip installation failed for {package}: {e}")
            run_cmd(f"!pip install {package}")

    # Lista dei pacchetti richiesti
    packages = [
        "numpy",
        "numpy<2.0,>=1.20",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "jovian --upgrade --quiet",
        "kaggle",
        "numpy.typing"
    ]

if not os.path.exists(r"C:\Users\manto\\"):

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
import os
import matplotlib.pyplot as plt
import surprise
import jovian
from sklearn.datasets import fetch_openml
import os
# %%
# if not os.path.exists("kaggle.json") and not os.path.exists("~/.kaggle/kaggle.json"):
if (False):
    ! mkdir ~/.kaggle
    ! cp kaggle.json ~/.kaggle/
    ! chmod 600 ~/.kaggle/kaggle.json
    ! kaggle datasets list 
    ! kaggle datasets download antonkozyriev/game-recommendations-on-steam 
    ! unzip game-recommendations-on-steam.zip -d games_dataset

# %% [markdown]
# ## Data cleaning
# %%
if os.path.exists(r"C:\Users\manto\\"):
    # print(os.getcwd())
    os.chdir(r"C:\Users\manto\Documents\Università\3°anno\Data_intensive\progetto-programmazione-di-applicazioni-data-intensive")
    print(os.getcwd())

# %%
# editing dataset dati aggiuntivi
if(True):
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
tagsdata['tags'] = tagsdata['tags'].dropna().str.replace(r"[ \[\]']", '', regex=True).str.split(',')
tagsdata['tags'].explode().nunique()


# %%
import csv

random.seed(42)
if(not os.path.exists('recommendations_half.csv')):
    with open('games_dataset/recommendations.csv', 'r') as inp, open('recommendations_half.csv', 'w') as out:
        writer = csv.writer(out)
        for i, row in enumerate(csv.reader(inp)):
            if random.random() < 0.5 or i == 0:  # ~50% di probabilità di tenere la riga
                writer.writerow(row)
# %%
rec_data = pd.read_csv("recommendations_half.csv", index_col=0)
rec_data=rec_data.join(tagsdata, on='app_id', how='inner')
if (True):
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
    rec_data['is_recommended'] = rec_data['is_recommended'].map(rank_stats)
    rec_data=rec_data.rename(columns={'is_recommended': 'rank'})
    rec_data['rank'] = rec_data['rank'].map(reduce3)
    rec_data=rec_data.loc[rec_data['user_id'].isin((rec_data['user_id'].value_counts()).iloc[:3000].index)]



# %%
rec_data.to_csv("games_dataset/tronc_rec.csv", encoding='utf-8')

# %%
if(True):
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

    # merge dei dataset
    tempdata=pd.merge(how='left', left=bigdata, right=secdata, left_index=True, right_index=True)
    data=pd.merge(how='left', left=tempdata, right=tagsdata, left_index=True, right_index=True)
    #data=pd.merge(how='left', left=data, right=rec_data, left_index=True, right_index=True)
    data.drop(columns='Tags', inplace=True)
    data.dropna(subset=['tags'], inplace=True)
# %%

# %% [markdown]
# %%
#grafico dei generi che hanno ricevuto piu' voti (torta e barre)
#grafico giochi piu' votati(data.head().pie)blabla
#media dei voti per genere
#numero di giochi che hanno un certo voto medio

# %%
# Ogni riga di 'tags' è una stringa di tag separati da virgole, quindi bisogna splittare, unire e trovare i valori unici
if(False):
    xGrafico=rec_data.join(tagsdata, on='app_id', how='left')
    rec_c = xGrafico.dropna(subset=['tags'])

# # %%
# pd.qcut(rec_c.groupby('user_id').size(), 10, duplicates='drop')#.value_counts().sort_index().plot.bar()

# # %%
# game_rew_bar=pd.qcut(rec_c.index.value_counts(), 10).value_counts().sort_index().plot.bar()
# plt.xlabel("Intervallo di recensioni per gioco")
# plt.ylabel("Numero di giochi")
# game_rew_bar.set_title("Distribuzione dei giochi per numero di recensioni")
# plt.show()
# # %%
# pd.qcut(xGrafico.index.value_counts(), 10).value_counts().sort_index().plot.pie()
# %%
#data=data.dropna(subset=['tags'])
# Estrai tutti i generi (tag) da ogni gioco, rimuovi parentesi quadre e spazi, splitta per virgola, poi esplodi in una serie
#all_tags = data['tags'].dropna().explode()
#Top 10 generi presenti nei videogiochi
#all_tags.value_counts().head(10).plot.bar()
#num_unique_tags = all_tags.unique()
#num_unique_tags[0]
# %%
#Top 10 generi meno presenti nei videogiochi
#all_tags.value_counts().tail(10).plot.bar()
# %%
# Esplodi la colonna tags
rec_tags = rec_data.dropna(subset=['tags']).explode('tags')

tag_counts = rec_tags['tags'].value_counts()

# Prendi i 10 tag più recensiti
top_tags = tag_counts.head(10).index

# Calcola la media rank solo per questi tag
top_tags_mean = rec_tags[rec_tags['tags'].isin(top_tags)].groupby('tags')['rank'].mean()

# Visualizza l'istogramma
# %%
top_tags_mean.sort_values(ascending=False, inplace=True)
top_tags_mean.plot.bar()
plt.xlabel("Tag")
plt.ylabel("Media rank")
plt.title("Top 10 tags per media rank nelle recensioni")
plt.show()
# %%
rec_data.drop(columns=['tags'], inplace=True)
# %%
tronc_data=rec_data.loc[rec_data['user_id'].isin((rec_data['user_id'].value_counts()).iloc[:2500].index)]
print(tronc_data.index.value_counts().index.nunique())
# %%
game_rew_bar=pd.qcut(tronc_data.index.value_counts(), 10, duplicates='drop').value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per gioco")
plt.ylabel("Numero di giochi")
game_rew_bar.set_title("Distribuzione dei giochi per numero di recensioni")
plt.show()
# %% [markdown]
# ## editing dataset dati aggiuntivi
# Alcune feature non sono rilevanti per il nostro problema, possiamo quindi rimuovere le colonne dal dataframe per risparmiare ulteriormente spazio.


