from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

class ArbreRegression:
    """Arbre de régression.
    
    Parameters
    ----------
    profondeur_maximale : ``int`` ou ``None`` (défaut ``None``)
        Profondeur maximale acceptable des arbres à construire.
        Si ``None``, l'arbre complet est construit.

    Example
    -------
    >>> modele = ArbreRegression(profondeur_maximale=12)
    """
    def __init__(self, profondeur_maximale=None):
        self._model = DecisionTreeRegressor(max_depth=profondeur_maximale)
    
    def entrainement(self, X, y):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste de valeurs réelles
            variable cible pour le jeu de données (=valeurs à prédire pour chaque individu)

        Example
        -------
        >>> modele = ArbreRegression(profondeur_maximale=...)
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
        >>> modele = ArbreRegression(profondeur_maximale=...)
        >>> modele.entrainement(X, y)
        >>> modele.prediction(X_new)
        """
        return self._model.predict(X)
    
    def evaluation(self, X, y):
        """Évalue les performances du modèle sur les données fournies.
        
        Calcule et retourne le coefficient de détermination R² et l'erreur quadratique moyenne (MSE).
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste de valeurs réelles
            variable cible pour le jeu de données (=valeurs à prédire pour chaque individu)

        Returns
        -------
        scores : dict
            Dictionnaire contenant :
            - 'r2' : float - Coefficient de détermination R²
            - 'mse' : float - Erreur quadratique moyenne (Mean Squared Error)

        Example
        -------
        >>> modele = ArbreRegression(profondeur_maximale=...)
        >>> modele.entrainement(X, y)
        >>> scores = modele.evaluation(X_new, y_new)
        >>> print(scores['r2'], scores['mse'])
        0.88 145.32
        """
        r2 = self._model.score(X, y)
        y_pred = self._model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return {'r2': r2, 'mse': mse}
    
class ForetRegression:
    """Forêt aléatoire pour la régression.
    
    Parameters
    ----------
    n_arbres : ``int`` (défaut 1)
        Nombre d'arbres à inclure dans la forêt
    profondeur_maximale : ``int`` ou ``None`` (défaut ``None``)
        Profondeur maximale acceptable des arbres à construire.
        Si ``None``, les arbres complets sont construits.

    Example
    -------
    >>> modele = ForetRegression(n_arbres=100, 
                                     profondeur_maximale=12)
    """
    def __init__(self, n_arbres=1, profondeur_maximale=None):
        self._model = RandomForestRegressor(n_estimators=n_arbres,
                                             max_depth=profondeur_maximale)
    
    def entrainement(self, X, y):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste de valeurs réelles
            variable cible pour le jeu de données (=valeurs à prédire pour chaque individu)

        Example
        -------
        >>> modele = ForetRegression(n_arbres=..., profondeur_maximale=...)
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
        >>> modele = ForetRegression(n_arbres=..., profondeur_maximale=...)
        >>> modele.entrainement(X, y)
        >>> modele.prediction(X_new)
        """
        return self._model.predict(X)
    
    def evaluation(self, X, y):
        """Évalue les performances du modèle sur les données fournies.
        
        Calcule et retourne le coefficient de détermination R² et l'erreur quadratique moyenne (MSE).
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données (variables explicatives)
        y : liste de valeurs réelles
            variable cible pour le jeu de données (=valeurs à prédire pour chaque individu)

        Returns
        -------
        scores : dict
            Dictionnaire contenant :
            - 'r2' : float - Coefficient de détermination R²
            - 'mse' : float - Erreur quadratique moyenne (Mean Squared Error)

        Example
        -------
        >>> modele = ForetRegression(n_arbres=..., profondeur_maximale=...)
        >>> modele.entrainement(X, y)
        >>> scores = modele.evaluation(X_new, y_new)
        >>> print(scores['r2'], scores['mse'])
        0.97 89.45
        """
        r2 = self._model.score(X, y)
        y_pred = self._model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return {'r2': r2, 'mse': mse}


class ClassificationAscendanteHierarchique:
    """Classification ascendante hiérarchique (CAH).
    
    Parameters
    ----------
    n_clusters : ``int`` (défaut 2)
        Nombre de clusters à former
    linkage : ``str`` (défaut 'simple')
        Type de lien : "simple", "complet" ou "moyen"

    Example
    -------
    >>> modele = ClassificationAscendanteHierarchique(n_clusters=3)
    """
    def __init__(self, n_clusters=2, linkage_method='simple'):
        d_linkage = {"simple": "single", "complet": "complete", "moyen": "average"}
        self._n_clusters = n_clusters
        self._linkage_method = d_linkage[linkage_method]
        self._linkage_matrix = None
        self._labels = None
    
    def entrainement(self, X):
        """Ajuste les paramètres du modèle sur les données fournies.
        
        Parameters
        ----------
        X : ``numpy.ndarray`` (~= liste de liste)
            jeu de données

        Example
        -------
        >>> modele = ClassificationAscendanteHierarchique(n_clusters=3)
        >>> modele.entrainement(X)
        """
        self._linkage_matrix = linkage(X, method=self._linkage_method)
        self._labels = fcluster(self._linkage_matrix, self._n_clusters, criterion='maxclust') - 1

    def labels(self):
        """Retourne les labels de cluster pour chaque point.
        
        Returns
        -------
        labels : ``numpy.ndarray``
            Label du cluster pour chaque point

        Example
        -------
        >>> modele = ClassificationAscendanteHierarchique(n_clusters=3)
        >>> modele.entrainement(X)
        >>> labels = modele.labels()
        """
        return self._labels
    
    def visualisation_dendrogramme(self):
        """Visualise la hiérarchie de clustering sous forme de dendrogramme.
        
        Example
        -------
        >>> modele = ClassificationAscendanteHierarchique(n_clusters=3)
        >>> modele.entrainement(X)
        >>> modele.visualisation_dendrogramme()
        """
        if self._linkage_matrix is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez entrainement() d'abord.")
        
        plt.figure(figsize=(10, 6))
        dendrogram(self._linkage_matrix)
        y = (self._linkage_matrix[-self._n_clusters + 1, 2] + self._linkage_matrix[-self._n_clusters, 2]) / 2
        plt.axhline(y=y, 
                    c='red', linestyle='--', label=f'Seuil pour {self._n_clusters} clusters')
        plt.xlabel('Index des échantillons')
        plt.ylabel('Distance')
        plt.title('Dendrogramme - Classification Ascendante Hiérarchique')
        plt.legend()
        plt.show()
