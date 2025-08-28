# %%
# INIZIO DEL NOTEBOOK - CONFIGURAZIONE INIZIALE E GESTIONE DEPENDENZE
# Questa cella è disattivata (if False), ma contiene uno script per installare automaticamente
# le librerie necessarie, con fallback da conda a pip, e gestione specifica per Windows.
# Utile per ambienti non configurati.

if False:
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
            try:
                run_cmd(f"conda install -y {package.split()[0]}")
                return
            except Exception:
                print(f"Conda installation failed for {package}, trying pip...")
        try:
            run_cmd(f"{sys.executable} -m pip install --upgrade pip")
            run_cmd(f"{sys.executable} -m pip install {package}")
        except Exception as e:
            print(f"Pip installation failed for {package}: {e}")
            run_cmd(f"!pip install {package}")

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
        run_cmd(f"{sys.executable} -m pip uninstall numpy -y")
        for pkg in packages:
            install(pkg)


# %%
# CARICAMENTO DELLE LIBRERIE PRINCIPALI
# Dopo l'installazione (eventuale), carichiamo tutte le librerie necessarie
# per l'analisi esplorativa, il preprocessing e la modellazione.

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import jovian
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import surprise
from datetime import datetime


# %%
# GESTIONE AMBIENTE E PATH
# Cambia la directory di lavoro in base all'ambiente in cui ci si trova
# (percorso Windows o Linux) per garantire che i file siano accessibili.

if os.path.exists("/home/ElManto03"):
    os.chdir("/home/ElManto03/Documenti/progetto-programmazione-di-applicazioni-data-intensive")
    print(os.getcwd())
if os.path.exists("/home/peppe"):
    os.chdir("/home/peppe/Documenti/progetto-programmazione-di-applicazioni-data-intensive")
    print(os.getcwd())


# %%
# CARICAMENTO E PULIZIA DEI DATI AGGIUNTIVI (secondo dataset da OpenML)
# Estraiamo informazioni aggiuntive sui giochi: Metacritic, Platform, Tags, Languages
# Rimuoviamo righe con valori mancanti e normalizziamo gli URL per ottenere l'app_id.

df = fetch_openml(data_id=43689)
secdata = pd.DataFrame({
    "SteamURL": df.data["SteamURL"],
    "Metacritic": df.data["Metacritic"],
    "Platform": df.data["Platform"],
    "Tags": df.data["Tags"],
    "Languages": df.data["Languages"]
})
secdata = secdata.dropna(axis='rows', subset=["SteamURL", "Tags"])
secdata['SteamURL'] = secdata['SteamURL'].str[35:-16]
secdata.drop_duplicates(subset=['SteamURL'], inplace=True)
secdata['SteamURL'] = secdata['SteamURL'].astype('int64')
secdata.index = secdata['SteamURL']
secdata.drop(columns=['SteamURL', 'Tags', 'Platform', 'Metacritic'], inplace=True)


# %%
# ENCODING ONE-HOT DELLE LINGUE
# Convertiamo la colonna 'Languages' in formato multilabel (lista di lingue)
# e usiamo MultiLabelBinarizer per creare colonne binarie per ogni lingua.

secdata['Languages'] = secdata['Languages'].dropna().str.replace(r"[ \[\]']", '', regex=True).str.split(',')
mlb = MultiLabelBinarizer()
multi_hot = mlb.fit_transform(secdata["Languages"])
secdata = secdata.join(pd.DataFrame(multi_hot, columns=mlb.classes_, index=secdata.index))
secdata.drop(columns=['Languages'], inplace=True)


# %%
# CARICAMENTO E PULIZIA DEI TAG (da metadata.json)
# Leggiamo i tag dai metadati dei giochi, li puliamo e trasformiamo in one-hot.

metadata = pd.read_json("games_dataset/games_metadata.json", lines="True")
metadata.to_csv("data3.csv", encoding='utf-8', index=False)
tagsdata = pd.read_csv("data3.csv", index_col=0)
tagsdata = tagsdata.drop(columns='description')
tagsdata.drop(tagsdata[tagsdata['tags'].str.len() == 2].index, inplace=True)  # rimuovi tag vuoti
tagsdata['tags'] = tagsdata['tags'].dropna().str.replace(r"[ \[\]']", '', regex=True).str.split(',')
tagsdata = tagsdata.loc[tagsdata.index.isin(secdata.index)]  # filtra per app_id in secdata
multi_hot = mlb.fit_transform(tagsdata['tags'])
tagsdata = tagsdata.join(pd.DataFrame(multi_hot, columns=mlb.classes_, index=tagsdata.index))
tagsdata.drop(columns='tags', inplace=True)


# %%
# RIDUZIONE DEL DATASET DI RECENSIONI
# Creiamo una versione più piccola del dataset delle raccomandazioni (~50%)
# per velocizzare gli esperimenti.

import csv
random.seed(42)
if not os.path.exists('recommendations_half.csv'):
    with open('games_dataset/recommendations.csv', 'r') as inp, open('recommendations_half.csv', 'w') as out:
        writer = csv.writer(out)
        for i, row in enumerate(csv.reader(inp)):
            if random.random() < 0.5 or i == 0:
                writer.writerow(row)


# %%
# CARICAMENTO DEL DATASET PRINCIPALE DI RECENSIONI
# Leggiamo il dataset ridotto e lo uniamo con i tag.

rec_data = pd.read_csv("recommendations_half.csv", index_col=0)
lightdata = rec_data.copy()
rec_data = rec_data.join(tagsdata, on='app_id', how='inner')


# %%
# CARICAMENTO DEL DATASET PRINCIPALE DEI GIOCHI (games.csv)
# Prepariamo il dataset principale: pulizia, encoding, normalizzazione.

bigdata = pd.read_csv("./games_dataset/games.csv", index_col=0)
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

bigdata["rating"] = bigdata["rating"].map(rank_stats_stream)
bigdata['win'] = bigdata['win'].astype(float)
bigdata['mac'] = bigdata['mac'].astype(float)
bigdata['linux'] = bigdata['linux'].astype(float)
bigdata.drop(columns=['title', 'user_reviews', 'discount', 'price_original'], inplace=True)


# %%
# FEATURE ENGINEERING: DATA DI RILASCIO E NORMALIZZAZIONE
# Estraiamo anno e mese, e normalizziamo alcune feature numeriche.

bigdata['date_release'] = pd.to_datetime(bigdata['date_release'])
bigdata["year"] = bigdata["date_release"].dt.year
bigdata["month"] = bigdata["date_release"].dt.month
bigdata.drop(columns="date_release", inplace=True)

scaler = MinMaxScaler()
num_features = scaler.fit_transform(bigdata[['year', 'month', 'price_final', 'positive_ratio']])
bigdata = bigdata.join(pd.DataFrame(num_features, columns=['year_std', 'month_std', 'price_final_std', 'positive_ratio_std'], index=bigdata.index))
bigdata.drop(columns=['year', 'month', 'price_final', 'positive_ratio'], inplace=True)
bigdata['rating'] = (bigdata['rating'] + 4) / 8  # normalizza rating tra 0 e 1


# %%
# MERGE FINALE DEI DATASET
# Uniamo bigdata, secdata e tagsdata in un unico dataframe completo.

tempdata = pd.merge(how='left', left=bigdata, right=secdata, left_index=True, right_index=True)
data = pd.merge(how='left', left=tempdata, right=tagsdata, left_index=True, right_index=True)


# %%
# ANALISI ESPLORATIVA: DISTRIBUZIONE UTENTI E GIOCHI
# Visualizziamo quante recensioni fanno gli utenti e quanti giochi sono recensiti.

rec_data['user_id'].value_counts()

user_rew_bar = pd.qcut(rec_data.groupby('user_id').size(), 10, duplicates='drop').value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per utente")
plt.ylabel("Numero di utenti")
user_rew_bar.set_title("Distribuzione degli utenti per numero di recensioni eseguite")
plt.show()


# %%
game_rew_bar = pd.qcut(rec_data.index.value_counts(), 10, duplicates='drop').value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per gioco")
plt.ylabel("Numero di giochi")
game_rew_bar.set_title("Distribuzione dei giochi per numero di recensioni")
plt.show()


# %%
# Filtriamo utenti molto attivi (>212 recensioni)
filtered_users = rec_data['user_id'].value_counts()
active_users = filtered_users[filtered_users > 212].index
filtered_rec_data = rec_data[rec_data['user_id'].isin(active_users)]
review_counts = filtered_rec_data['user_id'].value_counts()
review_bins = pd.qcut(review_counts, 10, duplicates='drop')
review_bins.value_counts().sort_index().plot.bar()
plt.xlabel("Intervallo di recensioni per utente (>212)")
plt.ylabel("Numero di utenti")
plt.title("Distribuzione utenti con >212 recensioni")
plt.show()


# %%
# Grafico a torta: distribuzione giochi per numero di recensioni
games_pie = pd.qcut(rec_data.index.value_counts(), 10, duplicates='drop').value_counts().sort_index(ascending=False).plot.pie(autopct='%1.1f%%', startangle=90)
games_pie.set_title("Distribuzione dei giochi per numero di recensioni")
games_pie.set_ylabel("")
plt.show()


# %%
# Top 10 tag più recensiti
rec_tags = rec_data.dropna(subset=['tags']).explode('tags')
tag_counts = rec_tags['tags'].value_counts()
top_tags = tag_counts.head(10).plot.bar()
plt.xlabel("Tag")
plt.ylabel("Numero recensioni")
plt.title("Top 10 tags per numero di recensioni")
plt.show()


# %%
# PREPARAZIONE DATI PER SURPRISE
datasurprise = rec_data.drop(columns=["helpful", "hours", "tags"])
datasurprise = datasurprise.reset_index()


# %%
from surprise import Reader, Dataset, model_selection, SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, accuracy, NormalPredictor
from surprise.accuracy import rmse, mae

df_reader = Reader(rating_scale=(1, 5))
datasurprise = Dataset.load_from_df(datasurprise[['user_id', 'app_id', 'rank']], df_reader)


# %%
# MODELLI SURPRISE: CROSS-VALIDATION
algo = KNNBasic()
model_selection.cross_validate(algo, datasurprise, verbose=True)


# %%
algo = NormalPredictor()
model_selection.cross_validate(algo, datasurprise, verbose=True)


# %%
algo = KNNWithMeans(k=300)
model_selection.cross_validate(algo, datasurprise, verbose=True)


# %%
# Train-test split e valutazione modelli
trainset, testset = surprise.model_selection.train_test_split(datasurprise, test_size=0.3)

model = SVD()
model.fit(trainset)
preds = model.test(testset)
print("SVD RMSE, MAE:", rmse(preds), mae(preds))


# %%
model_svdpp = SVDpp()
model_svdpp.fit(trainset)
preds_svdpp = model_svdpp.test(testset)
print("SVDpp RMSE:", accuracy.rmse(preds_svdpp))


# %%
model_nmf = NMF()
model_nmf.fit(trainset)
preds_nmf = model_nmf.test(testset)
print("NMF RMSE:", accuracy.rmse(preds_nmf))


# %%
# MATRICE FEATURES PER LIGHTFM
features_matrix = (bigdata.join(secdata, on="app_id", how="inner")).join(tagsdata, on="app_id", how="inner")


# %%
# Preparazione dati LightFM: solo recensioni positive (implicit feedback)
light_data = pd.read_csv("recommendations_half.csv", index_col=0)
light_data.drop(columns=['date', 'funny', 'review_id', 'helpful', 'hours'], inplace=True)
light_data.reset_index(inplace=True)
light_data = light_data.loc[light_data['is_recommended'] == True]
light_data = light_data.loc[light_data['app_id'].isin(features_matrix.index)]


# %%
# Campionamento utenti per fasce di attività
N = 12500
user_counts = light_data.groupby("user_id").size().reset_index(name="n_reviews")
bins = [0, 2, 5, 20, float("inf")]
labels = ["1-2", "3-5", "6-20", "20+"]
user_counts["fascia"] = pd.cut(user_counts["n_reviews"], bins=bins, labels=labels)

min_per_fascia = {"1-2": 3000, "3-5": 3000, "6-20": 3000, "20+": 500}
selected_users = []
for fascia, group in user_counts.groupby("fascia"):
    n_min = min_per_fascia.get(fascia, 0)
    n_take = min(len(group), n_min)
    sampled = group.sample(n=n_take, random_state=42)
    selected_users.append(sampled)
selected_users = pd.concat(selected_users)

remaining_slots = N - len(selected_users)
if remaining_slots > 0:
    remaining_pool = user_counts[~user_counts["user_id"].isin(selected_users["user_id"])]
    extra = remaining_pool.groupby("fascia").apply(
        lambda g: g.sample(n=min(len(g), int(remaining_slots * len(g) / len(remaining_pool))), random_state=42)
    ).reset_index(drop=True)
    selected_users = pd.concat([selected_users, extra])

light_data_red = light_data[light_data["user_id"].isin(selected_users["user_id"])]
features_matrix = features_matrix.loc[features_matrix.index.isin(light_data_red['app_id'].unique())]


# %%
# Preparazione finale dati LightFM
lightdata_m = light_data.loc[light_data['user_id'].isin(light_data['user_id'].value_counts().iloc[:7000].index)]
features_matrix = features_matrix.loc[features_matrix.index.isin(lightdata_m['app_id'].unique())]

lightdata = lightdata_m.copy()
lightdata['is_recommended'] = lightdata['is_recommended'].astype(int)
lightdata = lightdata.rename(columns={"app_id": "item_id", "is_recommended": "weight"})
lightdata = lightdata[['user_id', 'item_id', 'weight']]

lightdata['user_id'] = lightdata['user_id'].astype(str)
lightdata['item_id'] = lightdata['item_id'].astype(str)
lightdata['weight'] = lightdata['weight'].astype(float)

users = lightdata['user_id'].unique()
items = lightdata['item_id'].unique()

item_to_index = {item: idx for idx, item in enumerate(features_matrix.index.astype(str))}
items_sorted = sorted(items, key=lambda x: item_to_index[x])

features_matrix[['year_std', 'price_final_std']] += 1e-6
features_matrix.index = features_matrix.index.astype(str)


# %%
# CREAZIONE DATASET LIGHTFM
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split

dataset = Dataset()
dataset.fit(users, items, item_features=features_matrix.columns.tolist())

interactions, weights = dataset.build_interactions(lightdata.itertuples(index=False, name=None))
item_features = dataset.build_item_features(
    (item, features_matrix.columns[features_matrix.loc[item] > 0])
    for item in features_matrix.index
)
train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)


# %%
# TUNING IPERPARAMETRI LIGHTFM
param_grid = {
    "no_components": [10, 20, 50],
    "learning_rate": [0.01, 0.05, 0.1],
    "epochs": [20, 30, 50],
    "item_alpha": [1e-5, 1e-4, 1e-3]
}
results = []

for n in param_grid["no_components"]:
    for lr in param_grid["learning_rate"]:
        for ep in param_grid["epochs"]:
            for l2 in param_grid["item_alpha"]:
                model = LightFM(no_components=n, learning_rate=lr, loss="warp", random_state=42)
                model.fit(train, epochs=ep, num_threads=8, verbose=False, item_features=item_features)
                precision = precision_at_k(model, test, k=10, item_features=item_features, num_threads=8).mean()
                auc = auc_score(model, test, item_features=item_features, num_threads=8).mean()
                results.append({
                    "no_components": n, "lr": lr, "epochs": ep,
                    "precision": precision, "auc": auc, "item_alpha": l2
                })
                print(f"n={n}, lr={lr}, ep={ep}, l2={l2} -> prec@10={precision:.4f}, auc={auc:.4f}")

results_sorted = sorted(results, key=lambda x: x["precision"], reverse=True)
results_sorted_auc = sorted(results, key=lambda x: x["auc"], reverse=True)
print("Best precision:\n", results_sorted[0])
print("Best AUC:\n", results_sorted_auc[0])


# %%
# ADDESTRAMENTO MODELLO LIGHTFM FINALE
model = LightFM(no_components=20, loss="warp", item_alpha=1e-3, learning_rate=0.1)
model.fit(train, epochs=20, item_features=item_features, num_threads=4)


# %%
# VALUTAZIONE MODELLO
prec = precision_at_k(model, test, item_features=item_features, k=10, num_threads=4).mean()
auc = auc_score(model, test, item_features=item_features, num_threads=4).mean()
print(f"Precision@10: {prec:.4f} | AUC: {auc:.4f}")


# %%
# FUNZIONE DI RACCOMANDAZIONE TOP-K
def recommend_topk(model, dataset, interactions_csr, user_id, k=5):
    user_id_map, _, item_id_map, _ = dataset.mapping()
    u_internal = user_id_map[str(user_id)]
    item_labels = {v: k for k, v in item_id_map.items()}
    scores = model.predict(u_internal, np.arange(interactions_csr.shape[1]))
    seen = interactions_csr.tocsr()[u_internal].indices
    scores[seen] = -np.inf
    top_items = np.argsort(-scores)[:k]
    return [item_labels[i] for i in top_items]


# %%
# ESEMPIO DI RACCOMANDAZIONE PER 3 UTENTI CASUALI
for u in np.random.choice(users, 3, replace=False):
    raccomandazioni = recommend_topk(model, dataset, train, u, k=5)
    print(f"User {u} -> Recommended games: {raccomandazioni}")