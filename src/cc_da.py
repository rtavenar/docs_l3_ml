from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

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


    def visualisation_dendrogramme(self):
        """Visualise le dendrogramme correspondant à un modèle appris.

        Le dendrogramme permet de visualiser le process d'agglomération des individus.
        Il permet aussi de choisir le nombre de groupes car la hauteur des segments 
        correspond à la distance entre deux groupes.

        Example
        -------
        >>> modele = ClassificationAscendanteHierarchique()
        >>> modele.entrainement(X)
        >>> modele.visualisation_dendrogramme()
        """
        linkage_matrix = np.column_stack(
            [self._model.children_, self._model.distances_, np.zeros(self._model.children_.shape[0])]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(
            linkage_matrix, 
            color_threshold=0. if self._model.n_clusters is None else self._model.distances_[::-1][self._model.n_clusters-2],
            truncate_mode='lastp', p=10)
    

def charger_cifar():
    """Charge le jeu de données de classification d'images CIFAR10.
    
    Ce jeu de données contient des images d'objets et la tâche de classification
    consiste à retrouver le type d'objet contenu dans l'image (10 classes possibles).
    Chaque image est en résolution 28x28, ce qui fait 784 pixels en tout.

    Returns
    -------
    X_train : ``numpy.ndarray`` (~= liste de liste)
        jeu de données d'apprentissage (variables explicatives)
    y_train : liste d'entiers
        variable cible pour le jeu de données d'apprentissage (=classe)
    X_test : ``numpy.ndarray`` (~= liste de liste)
        jeu de données de test (variables explicatives)
    y_test : liste d'entiers
        variable cible pour le jeu de données de test (=classe)

    Example
    -------
    >>> X_app, X_test, y_app, y_test = charger_cifar()
    """
    loaded = np.load("cifar10.npz")
    X = loaded["X"]
    y = loaded["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    return X_train, X_test, y_train, y_test
