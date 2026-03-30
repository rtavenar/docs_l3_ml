from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans as sklearn_KMeans

class ArbreClassification:
    """Arbre binaire de classification.
    
    Parameters
    ----------
    profondeur_maximale : ``int`` ou ``None`` (défaut ``None``)
        Profondeur maximale acceptable des arbres à construire.
        Si ``None``, l'arbre complet est construit 
        (jusqu'à obtenir des feuilles pures).

    Example
    -------
    >>> modele = ArbreClassification(profondeur_maximale=12)
    """
    def __init__(self, profondeur_maximale=None):
        self._model = DecisionTreeClassifier(max_depth=profondeur_maximale)
    
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
        >>> modele = ArbreClassification(profondeur_maximale=...)
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
        >>> modele = ArbreClassification(profondeur_maximale=...)
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
        >>> modele = ArbreClassification(profondeur_maximale=...)
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
    profondeur_maximale : ``int`` ou ``None`` (défaut ``None``)
        Profondeur maximale acceptable des arbres à construire.
        Si ``None``, les arbres complets sont construits 
        (jusqu'à obtenir des feuilles pures).

    Example
    -------
    >>> modele = ForetClassification(n_arbres=100, 
                                     profondeur_maximale=12)
    """
    def __init__(self, n_arbres=1, profondeur_maximale=None):
        self._model = RandomForestClassifier(n_estimators=n_arbres,
                                             max_depth=profondeur_maximale)
    
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
        >>> modele = ForetClassification(n_arbres=..., profondeur_maximale=...)
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
        >>> modele = ForetClassification(n_arbres=..., profondeur_maximale=...)
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
        >>> modele = ForetClassification(n_arbres=..., profondeur_maximale=...)
        >>> modele.entrainement(X, y)
        >>> modele.evaluation(X_new, y_new)
        0.97
        """
        return self._model.score(X, y)


class KMeans:
    """Classification KMeans.

    Example
    -------
    >>> modele = KMeans(n_clusters=37)
    """
    def __init__(self, n_clusters):
        self._model = sklearn_KMeans(n_clusters=n_clusters, n_init=1)
    
    def entrainement(self, X):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données

        Example
        -------
        >>> modele = KMeans(n_clusters=37)
        >>> modele.entrainement(X)
        """
        self._model.fit(X)

    def inertie(self):
        """Retourne l'inertie intra-classe du modèle calculée sur le jeu d'apprentissage.

        Example
        -------
        >>> modele = KMeans(n_clusters=37)
        >>> modele.entrainement(X)
        >>> print(modele.inertie())
        """
        return self._model.inertia_
    
    def visualisation_clusters(self, X):
        """Visualise le jeu de données regroupé par clusters.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)

        Example
        -------
        >>> modele = KMeans(n_clusters=37)
        >>> modele.visualisation_clusters(X)
        """
        self.entrainement(X)

        plt.figure()
        for i_c in range(self._model.n_clusters):
            X_cluster = X[self._model.labels_ == i_c]
            plt.scatter(X_cluster[:, 0], X_cluster[:, 1])            
        plt.show()