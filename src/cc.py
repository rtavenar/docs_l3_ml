from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans as sklearn_KMeans

class ArbreClassification:
    """Arbre binaire de classification.
    
    Parameters
    ----------
    profondeur_max : ``int`` ou ``None`` (défaut ``None``)
        Profondeur maximale acceptable des arbres à construire.
        Si ``None``, l'arbre complet est construit 
        (jusqu'à obtenir des feuilles pures).

    Example
    -------
    >>> modele = ArbreClassification(profondeur_max=12)
    """
    def __init__(self, profondeur_max=None):
        self._model = DecisionTreeClassifier(max_depth=profondeur_max)
    
    def entrainement(self, X, y):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste d'entiers
            variable cible pour le jeu de données (=information de classe pour chaque individu)

        Example
        -------
        >>> modele = ArbreClassification(profondeur_max=...)
        >>> modele.entrainement(X, y)
        """
        self._model.fit(X, y)

    def prediction(self, X):
        """Calcule la prédiction du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)

        Example
        -------
        >>> modele = ArbreClassification(profondeur_max=...)
        >>> modele.entrainement(X, y)
        >>> modele.prediction(X_new)
        """
        return self._model.predict(X)
    
    def evaluation(self, X, y):
        """Évalue les performances (taux de bonnes classification) 
        du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste d'entiers
            variable cible pour le jeu de données (=information de classe pour chaque individu)

        Returns
        -------
        score : float
            Taux de bonnes classification obtenu par le modèle courant sur les données fournies

        Example
        -------
        >>> modele = ArbreClassification(profondeur_max=...)
        >>> modele.entrainement(X, y)
        >>> modele.evaluation(X_new, y_new)
        0.88
        """
        return self._model.score(X, y)
    
class ForetClassification:
    """Forêt aléatoire pour la classification.
    
    Parameters
    ----------
    n_arbres : ``int`` (défaut 1)
        Nombre d'arbres à inclure dans la forêt
    n_attributs_par_split : ``int`` (défaut 1)
        Nombre d'attributs à tirer au hasard pour chaque séparation dans 
        les arbres à construire
    profondeur_max : ``int`` ou ``None`` (défaut ``None``)
        Profondeur maximale acceptable des arbres à construire.
        Si ``None``, les arbres complets sont construits 
        (jusqu'à obtenir des feuilles pures).

    Example
    -------
    >>> modele = ForetClassification(n_arbres=100, 
                                     n_attributs_par_split=20, 
                                     profondeur_max=12)
    """
    def __init__(self, n_arbres=1, n_attributs_par_split=1, profondeur_max=None):
        self._model = RandomForestClassifier(n_estimators=n_arbres,
                                             max_features=n_attributs_par_split,
                                             max_depth=profondeur_max)
    
    def entrainement(self, X, y):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste d'entiers
            variable cible pour le jeu de données (=information de classe pour chaque individu)

        Example
        -------
        >>> modele = ForetClassification(n_arbres=..., n_attributs_par_split=..., profondeur_max=...)
        >>> modele.entrainement(X, y)
        """
        self._model.fit(X, y)

    def prediction(self, X):
        """Calcule la prédiction du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)

        Example
        -------
        >>> modele = ForetClassification(n_arbres=..., n_attributs_par_split=..., profondeur_max=...)
        >>> modele.entrainement(X, y)
        >>> modele.prediction(X_new)
        """
        return self._model.predict(X)
    
    def evaluation(self, X, y):
        """Évalue les performances (taux de bonnes classification) 
        du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste d'entiers
            variable cible pour le jeu de données (=information de classe pour chaque individu)

        Returns
        -------
        score : float
            Taux de bonnes classification obtenu par le modèle courant sur les données fournies

        Example
        -------
        >>> modele = ForetClassification(n_arbres=..., n_attributs_par_split=..., profondeur_max=...)
        >>> modele.entrainement(X, y)
        >>> modele.evaluation(X_new, y_new)
        0.97
        """
        return self._model.score(X, y)

class ClassificationAscendanteHierarchique:
    """Classification Ascendante Hierarchique.

    Example
    -------
    >>> modele = ClassificationAscendanteHierarchique()
    """
    def __init__(self):
        self._model = AgglomerativeClustering(compute_distances=True)
    
    def entrainement(self, X):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données

        Example
        -------
        >>> modele = ClassificationAscendanteHierarchique()
        >>> modele.entrainement(X)
        """
        self._model.fit(X)
    
    def visualisation_images(self, X, n_clusters):
        """Visualise quelques images du jeu de données regroupées par clusters.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste d'entiers
            variable cible pour le jeu de données (=information de classe pour chaque individu)

        Example
        -------
        >>> modele = ClassificationAscendanteHierarchique()
        >>> modele.entrainement(X)
        >>> modele.visualisation_images(X, n_clusters=...)
        """
        model = AgglomerativeClustering(compute_distances=True, n_clusters=n_clusters)
        model.fit(X)

        plt.figure(figsize=(4, n_clusters))
        for i_c in range(n_clusters):
            X_cluster = X[model.labels_ == i_c]
            indices = np.random.choice(len(X_cluster), size=4, replace=False)
            for i, idx in enumerate(indices):
                ax = plt.subplot(n_clusters, 4, i_c * 4 + i + 1)
                plt.imshow(X_cluster[idx].reshape((28, 28)), cmap="Greys")
                ax.set_xticks([])
                ax.set_yticks([])
            
        plt.tight_layout()
        plt.show()


class KMeans:
    """Classification KMeans.

    Example
    -------
    >>> modele = KMeans(k=37)
    """
    def __init__(self, k):
        self._model = sklearn_KMeans(n_clusters=k, n_init=1)
    
    def entrainement(self, X):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données

        Example
        -------
        >>> modele = KMeans(k=37)
        >>> modele.entrainement(X)
        """
        self._model.fit(X)

    def inertie(self):
        """Retourne l'inertie intra-classe du modèle calculée sur le jeu d'apprentissage.

        Example
        -------
        >>> modele = KMeans(k=37)
        >>> modele.entrainement(X)
        >>> print(modele.inertie())
        """
        return self._model.inertia_
    
    def visualisation_images(self, X):
        """Visualise quelques images du jeu de données regroupées par clusters.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste d'entiers
            variable cible pour le jeu de données (=information de classe pour chaque individu)

        Example
        -------
        >>> modele = KMeans(k=37)
        >>> modele.visualisation_images(X)
        """
        self.entrainement(X)

        plt.figure(figsize=(4, self._model.n_clusters))
        for i_c in range(self._model.n_clusters):
            X_cluster = X[self._model.labels_ == i_c]
            indices = np.random.choice(len(X_cluster), size=4, replace=False)
            for i, idx in enumerate(indices):
                ax = plt.subplot(self._model.n_clusters, 4, i_c * 4 + i + 1)
                plt.imshow(X_cluster[idx].reshape((28, 28)), cmap="Greys")
                ax.set_xticks([])
                ax.set_yticks([])
            
        plt.tight_layout()
        plt.show()
    

def charger_mnist():
    """Charge le jeu de données de classification d'images MNIST.
    
    Ce jeu de données contient des images de chiffres manuscrits et la tâche de classification
    consiste à retrouver le chiffre écrit dans l'image (10 classes possibles).
    Chaque image est en résolution 28x28, ce qui fait 784 pixels en tout.

    Returns
    -------
    X_train : ``numpy.ndarray`` (liste de liste)
        jeu de données d'apprentissage (variables explicatives)
    y_train : liste d'entiers
        variable cible pour le jeu de données d'apprentissage (=classe)
    X_test : ``numpy.ndarray`` (liste de liste)
        jeu de données de test (variables explicatives)
    y_test : liste d'entiers
        variable cible pour le jeu de données de test (=classe)

    Example
    -------
    >>> X_app, X_test, y_app, y_test = charger_mnist()
    >>> print(len(X_app))
    300
    >>> print(len(y_app))
    300
    >>> print(len(X_app[0]))
    784
    """
    loaded = np.load("mnist_subset.npz")
    X_train = loaded["images_apprentissage"]
    y_train = loaded["y_apprentissage"]
    X_test = loaded["images_test"]
    y_test = loaded["y_test"]
    return X_train, X_test, y_train, y_test


def visu_images(X, preds=None):
    """Visualise un jeu de données d'images de résolution 28x28

    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    y : liste d'entiers
        variable cible (=classe)
    preds : liste d'entiers, ou ``None``
        liste des prédictions fournies par un modèle. 
        Si ``None``, on ne visualise pas les informations 
        liées aux prédictions.

    Example
    -------
    >>> X_app, X_test, y_app, y_test = charger_mnist()
    >>> visu_images(X_app, y_app)
    >>> visu_images(X_test, y_test, modele.prediction(X_test))
    """
    np.random.seed(0)
    plt.figure(figsize=(8, 8))
    indices = np.random.choice(len(X), size=16, replace=False)
    for i, idx in enumerate(indices):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(X[idx].reshape((28, 28)), cmap="Greys")
        if preds is None:
            plt.title(f"Image {idx}\nClasse {y[idx]}")
        else:
            plt.title(f"Image {idx}\nClasse {y[idx]}, Prédite {preds[idx]}")
            if preds[idx] != y[idx]:
                plt.setp(ax.spines.values(), color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.show()
