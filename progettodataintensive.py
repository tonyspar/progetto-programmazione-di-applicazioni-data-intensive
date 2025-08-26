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

    if os.path.exists(r"C:\Users\manto\\"):

# Uninstall numpy first to avoid version conflicts
        run_cmd(f"{sys.executable} -m pip uninstall numpy -y")

        for pkg in packages:
            install(pkg)

# %% [markdown]
# ## 1. Descrizione del problema e analisi esplorativa
# ### Caricamento Librerie


# %%
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import jovian
from sklearn.datasets import fetch_openml
import os
import surprise
import lightfm
# %%
#if not os.path.exists("kaggle.json") and not os.path.exists("~/.kaggle/kaggle.json"):
""" if (False):
    ! mkdir ~/.kaggle
    ! cp kaggle.json ~/.kaggle/
    ! chmod 600 ~/.kaggle/kaggle.json
    ! kaggle datasets list 
    ! kaggle datasets download antonkozyriev/game-recommendations-on-steam 
    ! unzip game-recommendations-on-steam.zip -d games_dataset
 """# %%

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
    secdata.drop(columns=['SteamURL','Tags'], inplace=True)
    secdata
# %%
secdata
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
lightdata=rec_data.copy()
rec_data=rec_data.join(tagsdata, on='app_id', how='inner')

if (True):
    #ESTRAZIONE CASUALE IN BASE A VOTO METACRITIC
    random.seed(42)
    def rank_stats(value):
         if str(value).lower() == 'true':
            return np.random.choice([4, 5], p=[0.3, 0.7])
         else:
            return np.random.choice([1, 2], p=[0.7, 0.3])
    rec_data.columns
    rec_data.drop(columns=['date','funny','review_id'], inplace=True)
    rec_data['is_recommended'] = rec_data['is_recommended'].map(rank_stats)
    rec_data=rec_data.rename(columns={'is_recommended': 'rank'})
    rec_data=rec_data.loc[rec_data['user_id'].isin((rec_data['user_id'].value_counts()).iloc[:3000].index)]



# %%
#CSV GIÀ CREATO
#rec_data.to_csv("games_dataset/tronc_rec.csv", encoding='utf-8')

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

# %%
rec_data['user_id'].value_counts()

# %%
user_rew_bar=pd.qcut(rec_data.groupby('user_id').size(), 10, duplicates='drop').value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per utente")
plt.ylabel("Numero di utenti")
user_rew_bar.set_title("Distribuzione degli utenti per numero di recensioni eseguite")
plt.show()

# %%
game_rew_bar=pd.qcut(rec_data.index.value_counts(), 10, duplicates='drop').value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per gioco")
plt.ylabel("Numero di giochi")
game_rew_bar.set_title("Distribuzione dei giochi per numero di recensioni")
plt.show()
# %%
(rec_data['user_id'].value_counts()>1000).sum()

# %%
# Filtra solo gli utenti con più di 212 recensioni
filtered_users = rec_data['user_id'].value_counts()
active_users = filtered_users[filtered_users > 212].index
filtered_rec_data = rec_data[rec_data['user_id'].isin(active_users)]

# Conta le recensioni per utente tra quelli filtrati
review_counts = filtered_rec_data['user_id'].value_counts()

# Crea l'istogramma con pd.qcut
review_bins = pd.qcut(review_counts, 10, duplicates='drop')
review_bins.value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per utente (>212)")
plt.ylabel("Numero di utenti")
plt.title("Distribuzione utenti con >212 recensioni")
plt.show()

# %%
games_pie=pd.qcut(rec_data.index.value_counts(), 10, duplicates='drop').value_counts().sort_index(ascending=False).plot.pie(autopct='%1.1f%%', startangle=90)
games_pie.set_title("Distribuzione dei giochi per numero di recensioni")
games_pie.set_ylabel("")
# %%
data=data.dropna(subset=['tags'])
# Estrai tutti i generi (tag) da ogni gioco, rimuovi parentesi quadre e spazi, splitta per virgola, poi esplodi in una serie
all_tags = data['tags'].dropna().explode()
#Top 10 generi presenti nei videogiochi
genres_bar=all_tags.value_counts().head(10).plot.bar()
plt.xlabel("Generi più popolari")
plt.ylabel("Numero di giochi")
genres_bar.set_title("Numero di giochi per genere")
# %%
#Top 10 generi meno presenti nei videogiochi
#all_tags.value_counts().tail(10).plot.bar()
# %%

rec_data
# %%
# Esplodi la colonna tags
rec_tags = rec_data.dropna(subset=['tags']).explode('tags')

tag_counts = rec_tags['tags'].value_counts()

# Prendi i 10 tag più recensiti
top_tags = tag_counts.head(10).plot.bar()

# Visualizza l'istogramma
#top_tags_mean.sort_values(ascending=False, inplace=True)
plt.xlabel("Tag")
plt.ylabel("Numero recensioni")
plt.title("Top 10 tags per media rank nelle recensioni")
plt.show()
# %%
datasurprise=rec_data.drop(columns=["helpful", "hours", "tags"])

datasurprise

# %%
datasurprise=datasurprise.reset_index()
# %% [markdown]
# ## editing dataset dati aggiuntivi
# Alcune feature non sono rilevanti per il nostro problema, possiamo quindi rimuovere le colonne dal dataframe per risparmiare ulteriormente spazio.
# %%
#PER RECOMMENDATION SENZA MODELLO
#only_ratings = rec_data.pivot_table(values="rank", index="user_id", columns="app_id")
#only_ratings
(rec_data['rank'].value_counts().loc[1]+rec_data['rank'].value_counts().loc[2])/rec_data.shape[0]
# %%
from surprise import Reader, Dataset, model_selection, SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, accuracy, NormalPredictor
from surprise.accuracy import rmse, mae
from sklearn.metrics.pairwise import cosine_similarity
# %%
df_reader = Reader(rating_scale=(1,5))
datasurprise = Dataset.load_from_df(datasurprise[['user_id', 'app_id', 'rank']], df_reader) 
# %%
datasurprise
# %%
# carico il dataset
# definisco algoritmo da usare e eventuali parametri
#KNNBasic, non tocco tipo di similarità
algo = KNNBasic()

model_selection.cross_validate(algo, datasurprise, verbose=True)
# %%
algo = NormalPredictor()

# eseguo la cross validation a 5 fold
model_selection.cross_validate(algo, datasurprise, verbose=True)

# %%
#2°
algo = KNNWithMeans(k=300)

# eseguo la cross validation a 5 fold
model_selection.cross_validate(algo, datasurprise, verbose=True)
# %%
trainset, testset = surprise.model_selection.train_test_split(datasurprise, test_size=0.3)
if (False):
    algo = KNNBasic(k=300, sim_options={"user_based": False})
    
    algo.fit(trainset)
    
    preds = algo.test(testset)
    rmse(preds), mae(preds)
#model_selection.cross_validate(algo, datasurprise, verbose=True)
# %%
#1°
model = SVD()

model.fit(trainset)

preds = model.test(testset)
rmse(preds), mae(preds)

# %%
model_svdpp = SVDpp()
model_svdpp.fit(trainset)
preds_svdpp = model_svdpp.test(testset)
accuracy.rmse(preds_svdpp)
# %%
model_nmf = NMF()
model_nmf.fit(trainset)
preds_nmf = model_nmf.test(testset)
accuracy.rmse(preds_nmf)

# %% [markdown]
# Surprise - - Svd, SVD ++(top), NMF
# Bag of words con tags - ALS - CBF(sklearn)

# SVD
model_svd = SVD()
model_svd.fit(trainset)
preds_svd = model_svd.test(testset)
accuracy.rmse(preds_svd)

# SVD++
model_svdpp = SVDpp()
model_svdpp.fit(trainset)
preds_svdpp = model_svdpp.test(testset)
accuracy.rmse(preds_svdpp)

# NMF
model_nmf = NMF()
model_nmf.fit(trainset)
preds_nmf = model_nmf.test(testset)
accuracy.rmse(preds_nmf)

# %%
from lightfm.datasets import fetch_movielens
movielens = fetch_movielens()
for key, value in movielens.items():
    print(key, type(value), value.shape)
# %%
light_data = pd.read_csv("recommendations_half.csv", index_col=0)
light_data.drop(columns=['date','funny','review_id', "helpful", "hours"], inplace=True)
light_data.reset_index(inplace=True)
# %%
light_data=(light_data.loc[light_data['app_id'].isin(secdata.index)])
# %%
(light_data['user_id'].value_counts()>2).sum()
# %%
N = 4200  # Numero totale di utenti da selezionare
#FASCE A MANO


# Filtra il dataset originale per mantenere solo le recensioni degli utenti selezionati
light_data_reduced = light_data[light_data['user_id'].isin(sampled_users['user_id'])]

print(light_data_reduced['user_id'].nunique())  # Dovrebbe stampare 4200
# %%

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split

# --- 1) Usa i nomi reali delle tue colonne ---
# Se non hai un vero rating, aggiungi una colonna "rating" con 1.0
if 'rating' not in datasurprise.columns:
    datasurprise['rating'] = 1.0  # feedback implicito

# Converte a liste di tuple (utente, item, peso)
interactions_data = list(zip(
    datasurprise['user_id'].astype(str),
    datasurprise['item_id'].astype(str),
    datasurprise['rating'].astype(float)
))

users = datasurprise['user_id'].astype(str).unique()
items = datasurprise['item_id'].astype(str).unique()

# --- 2) Dataset + interazioni ---
dataset = Dataset()
dataset.fit(users, items)

interactions, _ = dataset.build_interactions(interactions_data)

train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)

# --- 3) Modello LightFM ---
model = LightFM(loss="warp", random_state=42)
model.fit(train, epochs=20, num_threads=4)

# --- 4) Valutazione ---
prec = precision_at_k(model, test, train_interactions=train, k=5, num_threads=4).mean()
auc = auc_score(model, test, train_interactions=train, num_threads=4).mean()
print(f"Precision@5: {prec:.3f} | AUC: {auc:.3f}")

# --- 5) Raccomandazioni ---
def recommend_topk(model, dataset, interactions_csr, user_id, k=5):
    # Mappature interne
    user_id_map, _, item_id_map, _ = dataset.mapping()
    u_internal = user_id_map[str(user_id)]
    item_labels = {v: k for k, v in item_id_map.items()}

    scores = model.predict(u_internal, np.arange(interactions_csr.shape[1]))

    seen = interactions_csr.tocsr()[u_internal].indices
    scores[seen] = -np.inf

    top_items = np.argsort(-scores)[:k]
    return [item_labels[i] for i in top_items]

for u in np.random.choice(users, 3, replace=False):
    print(f"{u} -> {recommend_topk(model, dataset, train, u, k=5)}")