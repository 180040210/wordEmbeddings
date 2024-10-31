import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

file = 'glove.6B.100d.txt'
word2vecModel = KeyedVectors.load_word2vec_format(file, binary=False, no_header=True)

# Task 1
inp = ['king', 'queen', 'man', 'woman', 'uncle', 'aunt', 'boy', 'girl']
print("Word Embeddings for specified inputs:")
for inp in inp:
    print(f"{inp} - Word2Vec Embedding: {word2vecModel[inp]}")

# Task 2:
analogies = [
    ('king', 'man', 'woman', 'queen'),
    ('uncle', 'man', 'woman', 'aunt'),
    ('boy', 'male', 'female', 'girl'),
    ('man', 'woman', 'sister', 'brother'),
    ('france', 'paris', 'rome', 'italy'),
    ('canada', 'ottawa', 'brasilia', 'brazil'),
    ('india', 'delhi', 'canberra', 'australia'),
    ('december', 'november', 'june', 'july'),
    ('august', 'february', 'october', 'april'),
    ('france', 'french', 'english', 'england')
]
print("\nVector Analogies:")
for wrd1, wrd2, wrd3, expected in analogies:
    out = word2vecModel.most_similar(positive=[wrd1, wrd3], negative=[wrd2])
    print(f"{wrd1} - {wrd2} + {wrd3} = {out[0][0]} (Expected is: {expected})")

# Task 3
checkWordsArray = ['automobile', 'ship', 'bike', 'man', 'woman', 'girl', 'boy', 'england', 'france', 'india', 'poland', 'february', 'september', 'december', 'paris', 'london', 'delhi']
print("\nMost Similar Words:")
for word in checkWordsArray:
    matchingWord = word2vecModel.most_similar(word, topn=5)
    print(f"{word} - Most similar words: {[w[0] for w in matchingWord]}")

# Task 4
def plot_tsne(words, model, title):
    vectors = np.array([model[word] for word in words])
    samples = vectors.shape[0]
    perplexity = min(30, samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.title(title)
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.show()

# first 100 words
words = list(word2vecModel.index_to_key[:100])
plot_tsne(words, word2vecModel, "t-SNE of First 100 Words")

# Clustering with KMeans
def plot_clusters(vectors, labels, title):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.scatter(vectors[:, 0], vectors[:, 1], c=labels, cmap='viridis')
    plt.show()

# KMeans clustering on first 100 words
first100Vectors = np.array([word2vecModel[word] for word in words])
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(first100Vectors)
plot_clusters(first100Vectors, kmeans_labels, "KMeans Clustering of First 100 Words")

# Gaussian Mixture Model clustering on first 100 words
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(first100Vectors)
plot_clusters(first100Vectors, gmm_labels, "GMM Clustering of First 100 Words")

# Plot last 100 words
last_100_words = list(word2vecModel.index_to_key[-100:])
plot_tsne(last_100_words, word2vecModel, "t-SNE of Last 100 Words")

# KMeans clustering on last 100 words
last_100_vectors = np.array([word2vecModel[word] for word in last_100_words])
kmeans_labels_last = kmeans.fit_predict(last_100_vectors)
plot_clusters(last_100_vectors, kmeans_labels_last, "KMeans Clustering of Last 100 Words")

# GMM clustering on last 100 words
gmm_labels_last = gmm.fit_predict(last_100_vectors)
plot_clusters(last_100_vectors, gmm_labels_last, "GMM Clustering of Last 100 Words")

# Similarity Visualization for Specific Words
specific_words = ['man', 'woman', 'automobile', 'computer', 'england', 'london', 'march']
for word in specific_words:
    similar_words = [w[0] for w in word2vecModel.most_similar(word, topn=10)]
    plot_tsne(similar_words, word2vecModel, f"t-SNE of words similar to '{word}'")
