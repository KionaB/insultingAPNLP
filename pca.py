import numpy as np
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Niet aanraken anders gaat alles kapot
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import torch 
import pickle

# Veel gedoe met cpu vs gpu, check wat er echt gebruikt wordt
if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"
print("Using device:", device)

model = SentenceTransformer("all-mpnet-base-v2", device=device)

animals = [
    "kitten", "puppy", "rabbit", "panda",
    "hamster", "dolphin", "horse",
    "fox", "deer", "owl", "cow", "chicken", "goat",
    "pig", "parrot", "rat", "snake",
    "bat", "vulture", "shark", "maggot", "worm", "hyena",
    # "lion", "tiger", "leopard", "cheetah", "jaguar", "elephant", "rhino",
    # "hippopotamus", "giraffe", "zebra", "kangaroo", "koala", "platypus",
    # "seal", "walrus", "beaver", "moose", "elk", "reindeer",
    # "penguin", "albatross", "seagull", "flamingo", "swan", "duck", "goose",
    # "turkey", "peacock", "toucan", "hummingbird", "woodpecker", "crow",
    # "raven", "sparrow", "finch", "canary", "owl", "eagle", "hawk",
    # "falcon", "condor", "parakeet", "macaw", "chameleon", "gecko", "iguana",
    # "turtle", "tortoise", "frog", "toad", "salamander", "newt", "crocodile",
    # "alligator", "lizard", "python", "cobra", "viper", "anaconda",
    # "centipede", "millipede", "bee", "wasp", "ant", "butterfly", "moth",
    # "dragonfly", "ladybug", "grasshopper", "lobster", "crab",
    # "shrimp", "octopus", "squid", "jellyfish", "starfish", "clam", "oyster",
    # "snail", "slug", "earthworm", "leeches", "mongoose", "weasel", "otter",
    # "badger", "skunk", "porcupine", "armadillo", "hedgehog", "rabbit",
    # "hare", "mouse", "voles", "chipmunk", "squirrel", "beetle",
    # "spider", "tarantula", "scorpion", "cricket", "cockroach"
]


# semantic axis of synonyms en antonyms
synonyms = ["friendly", "kind", "gentle", "nice"]
antonyms = ["hostile", "angry", "aggressive", "mean"]

# Doe de embeddings in een cache voor hergebruik bij herhaalde semantic axis
animal_cache_file = "animal_vecs.npy"
if os.path.exists(animal_cache_file):
    animal_vecs = np.load(animal_cache_file)
else:
    animal_vecs = model.encode(animals, convert_to_numpy=True, device=device)
    np.save(animal_cache_file, animal_vecs)

# Normaliseer elk dier individueel aangezien amplitude geen mening heeft.
animal_vecs = normalize(animal_vecs)

# Maak de PCA axis and return de direction vector, anchor mean, en hash scale.
def build_pca_axis(pos_words, neg_words, model):
    # Maak een unieke 'fingerafdruk' van je input om te zien of de semantic scale verandert
    # Dit kan ook een simpele check zijn of het semantic scale woord zelf veranderdt,
    # maar ik weet niet hoe synoiemen en antoniemen worden gemaakt op het moment (dit werkt sowieso)
    scale_hash = hash(tuple(pos_words) + tuple(neg_words))

    # Encode synonyms/antonyms woorden
    pos_vecs = normalize(model.encode(pos_words, convert_to_numpy=True, device=device))
    neg_vecs = normalize(model.encode(neg_words, convert_to_numpy=True, device=device))

    # Stack de twee matrixen op elkaar
    X = np.vstack([pos_vecs, neg_vecs])

    # doe de daadwerkelijke PCA berekening waarbij covariance matrx en richting van alle anchors worden berekend.
    pca = PCA(n_components=1)
    pca.fit(X)
    axis = pca.components_[0]
    axis /= np.linalg.norm(axis) # zorg opnieuw ervoor dat de PCA vector lengte 1 is.
    anchor_mean = X.mean(axis=0) # middenpunt van je anchors
    return axis, anchor_mean, scale_hash

# projecteer
def project_on_pca_axis(axis, anchor_mean, vec):
    centered = vec - anchor_mean # centreer PCA coordinated naar middenpunt van je anchor
    t = np.dot(centered, axis)
    proj_point = anchor_mean + t * axis
    d = np.linalg.norm(centered - t * axis)
    return t, d, proj_point

pca_cache_file = "pca_cache.pkl"
recompute_pca = True

# Check of PCA cache bestaat
if os.path.exists(pca_cache_file):
    with open(pca_cache_file, "rb") as f:
        cached_data = pickle.load(f) # laad de semantic scale hash, axis, en anchor mean
    cached_hash = cached_data["scale_hash"]

    if cached_hash == hash(tuple(synonyms) + tuple(antonyms)): #Still the same combination of synonyms and antonyms?
        axis = cached_data["axis"]
        anchor_mean = cached_data["anchor_mean"]
        recompute_pca = False

if recompute_pca: #Only calculate CPA if not cached 
    axis, anchor_mean, scale_hash = build_pca_axis(synonyms, antonyms, model)
    with open(pca_cache_file, "wb") as f:
        pickle.dump({"axis": axis, "anchor_mean": anchor_mean, "scale_hash": scale_hash}, f) # Save new PCA values and hash scale

results = []
for animal, vec in zip(animals, animal_vecs):
    t, d, _ = project_on_pca_axis(axis, anchor_mean, vec)
    results.append((animal, t, d))

results.sort(key=lambda x: x[1])

# Print table
table = [[w, f"{t:.3f}", f"{d:.3f}"] for w, t, d in results]
print(tabulate(table, headers=["Word", "t (PC1 position)", "Orthogonal distance"], tablefmt="fancy_grid"))