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
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import surprise
import lightfm
from datetime import datetime

# %%
#if not os.path.exists("kaggle.json") and not os.path.exists("~/.kaggle/kaggle.json"):
"""  if (True):
    ! mkdir ~/.kaggle
    ! cp kaggle.json ~/.kaggle/
    ! chmod 600 ~/.kaggle/kaggle.json
    ! kaggle datasets list 
    ! kaggle datasets download antonkozyriev/game-recommendations-on-steam 
    ! unzip game-recommendations-on-steam.zip -d games_dataset """
 # %%
# %
# %% [markdown]
# ## Data cleaning
# %%
#if os.path.exists(r"C:\Users\manto\\"):
    # print(os.getcwd())
    #os.chdir(r"C:\Users\manto\Documents\Università\3°anno\Data_intensive\progetto-programmazione-di-applicazioni-data-intensive")
if os.path.exists("/home/ElManto03"):
    os.chdir("/home/ElManto03/Documenti/progetto-programmazione-di-applicazioni-data-intensive")
    print(os.getcwd())
if os.path.exists("/home/peppe"):
    os.chdir("/home/peppe/Documenti/progetto-programmazione-di-applicazioni-data-intensive")
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
    secdata.drop(columns=['SteamURL','Tags', 'Platform', 'Metacritic'], inplace=True)

# %%
secdata
secdata['Languages'] = secdata['Languages'].dropna().str.replace(r"[ \[\]']", '', regex=True).str.split(',')
secdata['Languages'].explode().nunique()
# %%
mlb = MultiLabelBinarizer()

multi_hot = mlb.fit_transform(secdata["Languages"])
secdata = secdata.join(pd.DataFrame(multi_hot, columns=mlb.classes_, index=secdata.index))
secdata.drop(columns=['Languages'], inplace=True)
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
tagsdata=tagsdata.loc[tagsdata.index.isin(secdata.index)]
multi_hot = mlb.fit_transform(tagsdata['tags'])
tagsdata=tagsdata.join(pd.DataFrame(multi_hot, columns=mlb.classes_,index=tagsdata.index))
tagsdata.drop(columns='tags', inplace=True)
tagsdata
#rec_data=rec_data.loc[rec_data['user_id'].isin((rec_data['user_id'].value_counts()).iloc[:3000].index)]
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
    #rec_data=rec_data.loc[rec_data['user_id'].isin((rec_data['user_id'].value_counts()).iloc[:3000].index)]



# %%
#CSV GIÀ CREATO
#rec_data.to_csv("games_dataset/tronc_rec.csv", encoding='utf-8')

# %%
if(True):
    bigdata=pd.read_csv("./games_dataset/games.csv", index_col=0)
    #bigdata['steam_deck'].value_counts()
    bigdata.drop(columns=["steam_deck"], inplace=True)
    #def rank_stats_stream(value):
    #    mapping = {
    #        'overwhelmingly positive': 9-5,
    #        'very positive': 8-5,
    #        'mostly positive': 7-5,
    #        'positive': 6-5,
    #        'mixed': 5-5,
    #        'negative': 4-5,
    #        'mostly negative': 3-5,
    #        'overwhelmingly negative': 1-5,
    #        'very negative': 2-5
    #    }
    #    return mapping.get(str(value).lower(), np.nan)

    #bigdata["rating"]= bigdata["rating"].map(rank_stats_stream)
    bigdata['rating'].unique()
    #rec_data['is_recommended'] = rec_data['is_recommended'].map(rank_stats)

    # merge dei dataset
    #tempdata=pd.merge(how='left', left=bigdata, right=secdata, left_index=True, right_index=True)
    #data=pd.merge(how='left', left=tempdata, right=tagsdata, left_index=True, right_index=True)
    #data=pd.merge(how='left', left=data, right=rec_data, left_index=True, right_index=True)
bigdata.drop(columns='rating', inplace=True)
bigdata
# %%
names_data=bigdata[['title']]
names_data
# %%
bigdata['win']=bigdata['win'].astype(float)
bigdata['mac']=bigdata['mac'].astype(float)
bigdata['linux']=bigdata['linux'].astype(float)

# %%
bigdata.drop(columns=['title', 'user_reviews', 'discount', 'price_original'], inplace=True)
# %%
#bigdata
bigdata['date_release']=pd.to_datetime(bigdata['date_release'])
bigdata["year"] = bigdata["date_release"].dt.year
bigdata["month"] = bigdata["date_release"].dt.month
bigdata.drop(columns="date_release", inplace=True)
# %%
rec_data
# %%
scaler= MinMaxScaler()
num_features=scaler.fit_transform(bigdata[['year', 'month', 'price_final', 'positive_ratio']])
num_features[:5]
# %%
bigdata_std=bigdata.join(pd.DataFrame(num_features, columns=['year_std', 'month_std', 'price_final_std', 'positive_ratio_std'], index=bigdata.index))
bigdata_std.drop(columns=['year', 'month', 'price_final', 'positive_ratio'], inplace=True)
# %%
bigdata
# %% [markdown]
# Tags con esplosione in colonne
# piattaforma con one-hot encode
# Metacritic, positive_ratio, date-release, price_final, price_original, rating
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
rec_data

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
if(False):
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

# %%
# GRAFICO: Distribuzione delle piattaforme supportate
# Mostra quanti giochi supportano Windows, macOS e Linux

platform_counts = {
    'Windows': bigdata['win'].sum(),
    'macOS': bigdata['mac'].sum(),
    'Linux': bigdata['linux'].sum()
}

plt.figure(figsize=(8, 5))
plt.bar(platform_counts.keys(), platform_counts.values(), color=['#0066cc', '#ff9900', '#33cc33'])
plt.title("Numero di giochi per piattaforma supportata")
plt.ylabel("Numero di giochi")
plt.xlabel("Piattaforma")
for i, v in enumerate(platform_counts.values()):
    plt.text(i, v + 100, str(int(v)), ha='center', va='bottom')
plt.show()

# %%
# GRAFICO: Prezzo dei giochi - Fasce in euro
# Crea fasce di prezzo e mostra distribuzione con torta e istogramma

bigdata_price = bigdata.copy()
#bigdata_price['price_final'] = bigdata_price['price_final'] * (bigdata['price_final_std'].max() - bigdata['price_final_std'].min()) + bigdata['price_final_std'].min()
#
bins = [0, 10, 30, 80, float('inf')]
labels = ['0-10€', '10-30€', '30-80€', '>80€']
bigdata_price['price_bin'] = pd.cut(bigdata_price['price_final'], bins=bins, labels=labels, right=False)

price_counts = bigdata_price['price_bin'].value_counts().sort_index()

# Grafico a torta
plt.figure(figsize=(12, 5))

# Istogramma
plt.subplot(1, 2, 2)
price_counts.plot.bar(color=plt.cm.Paired(range(len(price_counts))))
plt.title("Numero di giochi per fasce di prezzo")
plt.ylabel("Numero giochi")
plt.xlabel("Fascia di prezzo")
for i, v in enumerate(price_counts):
    plt.text(i, v + 50, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# %%
# GRAFICO: Distribuzione del positive_ratio
# Quanto sono positivi i voti medi dei giochi

bigdata_ratio = bigdata.copy()
#bigdata_ratio['positive_ratio'] = bigdata_ratio['positive_ratio_std'] * (bigdata['positive_ratio'].max() - bigdata['positive_ratio'].min()) + bigdata['positive_ratio'].min()

plt.figure(figsize=(10, 6))
plt.hist(bigdata_ratio['positive_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Distribuzione del Positive Ratio (percentuale recensioni positive)")
plt.xlabel("Positive Ratio (%)")
plt.ylabel("Numero di giochi")
plt.grid(axis='y', alpha=0.3)
plt.show()

# %%
# GRAFICO: Numero di giochi rilasciati per mese
# Mostra in quale mese dell'anno escono più giochi

# Assicurati che 'month' esista
if 'month' not in bigdata.columns:
    bigdata_temp = pd.read_csv("./games_dataset/games.csv", index_col=0)
    bigdata_temp['date_release'] = pd.to_datetime(bigdata_temp['date_release'])
    months = bigdata_temp['date_release'].dt.month
else:
    months = bigdata['month']

plt.figure(figsize=(10, 6))
months.plot.hist(bins=12, color='coral', alpha=0.7, edgecolor='black')
plt.title("Distribuzione dei giochi per mese di rilascio")
plt.xlabel("Mese")
plt.ylabel("Numero di giochi")
plt.xticks(range(1, 13))
plt.grid(axis='y', alpha=0.3)
plt.show()

# %%
# GRAFICO: Positive Ratio vs Prezzo
# Esiste una correlazione tra prezzo e qualità percepita?

bigdata_plot = bigdata.copy()
# Riportiamo i valori standardizzati ai valori originali

plt.figure(figsize=(10, 6))
plt.scatter(bigdata_plot['price_final'], bigdata_plot['positive_ratio'], alpha=0.5, color='teal')
plt.title("Positive Ratio vs Prezzo del gioco")
plt.xlabel("Prezzo (€)")
plt.ylabel("Positive Ratio (%)")
plt.grid(True, alpha=0.3)
plt.show()


# Correlazione
corr = bigdata_plot['price_final'].corr(bigdata_plot['positive_ratio'])
print(f"Correlazione tra prezzo e positive ratio: {corr:.3f}")

# %%
# GRAFICO: Numero di recensioni per gioco vs Prezzo
# I giochi più costosi ricevono più recensioni?

# Conta quante recensioni ha ogni gioco (app_id)
reviews_per_game = rec_data.reset_index()['app_id'].value_counts()
reviews_per_game
# %%

# Unisci con il prezzo
price_reviews = pd.DataFrame({
    'price_final': bigdata['price_final'],
}).join(reviews_per_game, on='app_id', how='inner').rename(columns={'app_id': 'review_count'})

price_reviews
# %%
# Rimuovi outliers per una migliore visualizzazione
price_reviews = price_reviews[price_reviews['count'] < 5000]  # filtro estremi

plt.figure(figsize=(10, 6))
plt.scatter(price_reviews['price_final'], price_reviews['count'], alpha=0.5, color='purple')
plt.title("Numero di recensioni per gioco vs Prezzo")
plt.xlabel("Prezzo (€)")
plt.ylabel("Numero di recensioni")
plt.grid(True, alpha=0.3)
plt.show()

# Correlazione
corr_reviews_price = price_reviews['price_final'].corr(price_reviews['count'])
print(f"Correlazione tra prezzo e numero di recensioni: {corr_reviews_price:.3f}")


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
rec_data=rec_data.loc[rec_data['user_id'].isin((rec_data['user_id'].value_counts()).iloc[:3000].index)]
datasurprise=rec_data.drop(columns=["helpful", "hours"]) #tags

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
from surprise.model_selection import GridSearchCV
# %%
df_reader = Reader(rating_scale=(1,5))
datasurprise = Dataset.load_from_df(datasurprise[['user_id', 'app_id', 'rank']], df_reader) 
# %%
datasurprise
# %%
param_grid = {
    "k": [50, 100, 300],
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'], 
        'user_based': [True, False]}
}
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

gs.fit(datasurprise)

# Migliori parametri trovati
print("Migliori parametri (RMSE):", gs.best_params['rmse'])
print("Miglior RMSE:", gs.best_score['rmse'])
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
# MATRICE FEATURES
features_matrix=(bigdata.join(secdata, on="app_id", how="inner")).join(tagsdata, on="app_id", how="inner")
features_matrix
# %%
light_data = pd.read_csv("recommendations_half.csv", index_col=0)
light_data.drop(columns=['date','funny','review_id', "helpful", "hours"], inplace=True)
light_data.reset_index(inplace=True)
# Elimino i dati con recensioni negative perché uso una recommendation implicita
light_data=light_data.loc[light_data['is_recommended']==True]
light_data=(light_data.loc[light_data['app_id'].isin(features_matrix.index)])

# %%
print(light_data['user_id'].nunique())
print(light_data['app_id'].nunique())
light_data
# %%
N = 12500  # Numero totale di utenti da selezionare
#FASCE A MANO
user_counts = light_data.groupby("user_id").size().reset_index(name="n_reviews")

# Definisci fasce manuali
bins = [0, 2, 5, 20, float("inf")]
# 6.5m, 1m, 400k, 40k
labels = ["1-2", "3-5", "6-20", "20+"]
user_counts["fascia"] = pd.cut(user_counts["n_reviews"], bins=bins, labels=labels)
user_counts["fascia"].value_counts()
# Quanti utenti vogliamo in totale 

# Minimo per fascia (puoi aggiustare tu questi valori)
min_per_fascia = {
     "1-2": 3000,
    "3-5": 3000,
    "6-20": 3000,
    "20+": 500   # i super attivi sono pochi, bastano meno
}

# --- 1) campiona il minimo garantito ---
selected_users = []
for fascia, group in user_counts.groupby("fascia"):
    n_min = min_per_fascia.get(fascia, 0)
    n_take = min(len(group), n_min)
    sampled = group.sample(n=n_take, random_state=42)
    selected_users.append(sampled)

selected_users = pd.concat(selected_users)

# --- 2) se non raggiungi N, riempi proporzionalmente ---
remaining_slots = N - len(selected_users)
if remaining_slots > 0:
    # prendi il resto proporzionalmente dalle fasce, esclusi quelli già scelti
    remaining_pool = user_counts[~user_counts["user_id"].isin(selected_users["user_id"])]
    extra = (
        remaining_pool.groupby("fascia")
        .apply(lambda g: g.sample(
            n=min(len(g), int(remaining_slots * len(g) / len(remaining_pool))),
            random_state=42
        ))
        .reset_index(drop=True)
    )
    selected_users = pd.concat([selected_users, extra])

# --- 3) Filtra il dataset originale ---
light_data_red = light_data[light_data["user_id"].isin(selected_users["user_id"])]
# %%
features_matrix=(features_matrix.loc[features_matrix.index.isin(light_data_red['app_id'].unique())])
print(features_matrix.index.nunique())

print("Utenti selezionati:", light_data_red["user_id"].nunique())
print("Recensioni rimaste:", len(light_data_red))
# %%
type(light_data_red['app_id'].unique()[0])

# %%
light_data_red
# %%
lightdata_m=light_data.loc[light_data['user_id'].isin((light_data['user_id'].value_counts()).iloc[:7000].index)]
print(lightdata_m.shape[0])
print(lightdata_m['app_id'].nunique())
# %%
features_matrix=(features_matrix.loc[features_matrix.index.isin(lightdata_m['app_id'].unique())])
print(features_matrix.index.nunique())
# %%

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split    

# ----------------------------------------------------
# 1) PREPARAZIONE DEI DATI DI PARTENZA
# ----------------------------------------------------
# lightdata è il DataFrame su cui lavoriamo.
# Deve contenere almeno:
#   - una colonna con l'ID utente (es. 'user_id')
#   - una colonna con l'ID item/prodotto (es. 'item_id')
#   - opzionalmente un 'rating' o 'peso' dell'interazione
#
# Se il dataset rappresenta feedback impliciti
# (es. click, acquisti, visualizzazioni)
# e NON contiene una colonna 'rating', ne creiamo una fittizia
# impostando il punteggio a 1.0 per ogni riga.
# Questo indica che "utente X ha interagito con item Y"
# senza indicare un grado di preferenza.

# lightdata = il tuo DataFrame in formato wide come quello mostrato
# Indice = user_id, colonne = app_id, valori booleani o NaN
lightdata=lightdata_m

# 1. Sostituire i NaN con 0 e i booleani con 1
lightdata['is_recommended'] = lightdata['is_recommended'].astype(int)
# Se hai già True/False, puoi fare: binary_df = lightdata.astype(int)
lightdata=lightdata.rename(columns={"app_id" : "item_id", "is_recommended" : "weight"})
lightdata = lightdata[['user_id', 'item_id', 'weight']]
lightdata
# %%
# Assicurati di avere user_id, item_id e rating

# Conversione tipi (buona pratica)
lightdata['user_id'] = lightdata['user_id'].astype(str)
lightdata['item_id'] = lightdata['item_id'].astype(str)
lightdata['weight'] = lightdata['weight'].astype(float)



# Ricaviamo anche l'elenco di utenti e item univoci.
# Saranno utili per "istruire" il Dataset sul vocabolario completo.
users = lightdata['user_id'].unique()
items = lightdata['item_id'].unique()
#  %%
item_to_index = {item: idx for idx, item in enumerate((features_matrix.index).astype(str))}

# ordina items_filtered secondo feature_items
items_sorted = sorted(items, key=lambda x: item_to_index[x])
# %%
features_matrix[['year_std', 'price_final_std']] = features_matrix[['year_std', 'price_final_std']]+1e-6

# %%
features_matrix.index = features_matrix.index.astype(str)
# %%
features_matrix
#lightdata
#users
#
# %%
# ----------------------------------------------------
# 2) CREAZIONE DEL DATASET E MATRICE DI INTERAZIONI
# ----------------------------------------------------
# Istanzio un oggetto Dataset che si occuperà di mappare
# i miei ID testuali (user/item) a indici interi interni.
dataset = Dataset()
dataset.fit(users, items, item_features=features_matrix.columns.tolist())

# Costruisco la matrice delle interazioni (sparsa CSR)
# a partire dalle tuple utente-item-rating.
interactions, weights = dataset.build_interactions(lightdata.itertuples(index=False, name=None))

item_features = dataset.build_item_features((item, features_matrix.columns[features_matrix.loc[item] > 0]) 
     for item in features_matrix.index)

# Suddivido in train e test in modo casuale,
# mantenendo la distribuzione di interazioni (sparsità).
train, test = random_train_test_split(
    interactions, test_percentage=0.2, random_state=42
)
# %%
# RICERCA IPERPARAMETRI PER IL MODELLO
param_grid = {
    "no_components": [10, 20, 50, 100],
    "learning_schedule": ["adagrad", "adadelta"],
    "learning_rate": [0.01, 0.05, 0.1],
    "epochs": [5, 10, 20]
    #"item_alpha": [1e-6, 1e-8]
}
results = []

for n in param_grid["no_components"]:
    for sched in param_grid["learning_schedule"]:
        for lr in param_grid["learning_rate"]:
            for ep in param_grid["epochs"]:
                
                model = LightFM(
                    no_components=n,
                    learning_schedule=sched,
                    learning_rate=lr,
                    loss="warp",   # tieni una loss fissa
                    random_state=42
                )
                
                model.fit(train, epochs=ep, num_threads=4, verbose=False)
                
                # valutazione
                precision = precision_at_k(model, test, k=10).mean()
                auc = auc_score(model, test).mean()
                
                results.append({
                    "no_components": n,
                    "schedule": sched,
                    "lr": lr,
                    "epochs": ep,
                    "precision": precision,
                    "auc": auc
                })
                
                print(f"n={n}, sched={sched}, lr={lr}, ep={ep} -> prec@10={precision:.4f}, auc={auc:.4f}")
# %%
#2° RICERCA IPERPARAMETRI

param_grid = {
    "no_components": [10, 20, 50],#100
    "learning_rate": [0.01, 0.05, 0.1],
    "epochs": [20, 30, 50],#100
    "item_alpha": [1e-5, 1e-4, 1e-3]
}

results = []

for n in param_grid["no_components"]:
    for lr in param_grid["learning_rate"]:
        for ep in param_grid["epochs"]:
            for l2 in param_grid['item_alpha']:

                model = LightFM(
                    no_components=n,
                    learning_rate=lr,
                    loss="warp",      # tieni la stessa loss
                    random_state=42,
                    item_alpha=l2
                )

                model.fit(train, epochs=ep, num_threads=8, verbose=False, item_features=item_features)

                # valutazione
                precision = precision_at_k(model, test, k=10, item_features=item_features, num_threads=8).mean()
                auc = auc_score(model, test, item_features=item_features, num_threads=8).mean()

                results.append({
                    "no_components": n,
                    "lr": lr,
                    "epochs": ep,
                    "precision": precision,
                    "auc": auc,
                    "item_alpha": l2
                })

                print(f"n={n}, lr={lr}, ep={ep} ,l2={l2} -> prec@10={precision:.4f}, auc={auc:.4f}")
                



# %%
# ordina per la metrica che preferisci
results_sorted = sorted(results, key=lambda x: x["precision"], reverse=True)
results_sorted_auc = sorted(results, key=lambda x: x["auc"], reverse=True)

#print("Miglior combinazione trovata:")
print("Best precision:\n")
print(results_sorted[0])
print("Best AUC:\n")
print(results_sorted_auc[0])
# %%
# ----------------------------------------------------
# 3) CREAZIONE E ADDESTRAMENTO DEL MODELLO
# ----------------------------------------------------
# loss='warp' = Weighted Approximate-Rank Pairwise:
# adatto a ottimizzare il ranking top‑N con feedback implicito.
# cambia n_components (maybe)
#provare con learning_schedule adadelta
#cambiare learning rate
#rho e epsilon???
#migliori parametri finora
model = LightFM(no_components=100, loss="warp", item_alpha=1e-4, learning_rate=0.1)
#cambia numero di epoche
model.fit(train, epochs=100, item_features=item_features, num_threads=8)
# %%
# ----------------------------------------------------
# 4) VALUTAZIONE DEL MODELLO
# ----------------------------------------------------
# Precision@K: quanta parte dei primi K suggerimenti è rilevante.
prec = precision_at_k(model, test, item_features=item_features, k=10, num_threads=8).mean()

# AUC (Area Under Curve): probabilità che un item rilevante
# abbia punteggio maggiore di uno irrilevante.
auc = auc_score(model, test, item_features=item_features, num_threads=8).mean()

print(f"Precision@10: {prec} | AUC: {auc:.3f}")
# %%
features_matrix.columns
(model.item_embeddings.mean(axis=1)<1e-3).sum()
# %%
# ----------------------------------------------------
# 5) FUNZIONE PER RACCOMANDARE TOP-K ITEM
# ----------------------------------------------------
def recommend_topk(model, dataset, interactions_csr, user_id, k=5):#qua va feature?
    # Mappature interne da stringhe a indici interi
    user_id_map, _, item_id_map, _ = dataset.mapping()
    u_internal = user_id_map[str(user_id)]
    item_labels = {v: k for k, v in item_id_map.items()}

    # Predico uno score per OGNI item rispetto all'utente specificato
    scores = model.predict(u_internal, np.arange(interactions_csr.shape[1]))

    # Escludo gli item che l'utente ha già visto/interagito
    seen = interactions_csr.tocsr()[u_internal].indices
    scores[seen] = -np.inf  # azzero la possibilità di raccomandarli

    # Ordino gli item per punteggio decrescente e prendo i primi K
    top_items = np.argsort(-scores)[:k]
    return [item_labels[i] for i in top_items]

# ----------------------------------------------------
# 6) ESEMPIO: RACCOMANDAZIONI PER 3 UTENTI CASUALI
# ----------------------------------------------------
for u in np.random.choice(users, 3, replace=False):
    raccomandazioni = recommend_topk(model, dataset, train, u, k=5)
    print(f"{u} -> {raccomandazioni}")

# %%
