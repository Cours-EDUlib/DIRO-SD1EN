---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Démonstration 2 - IFT 3700/6758
### Sujets abordés :
- Présentation de la librairie scikit-learn
- k-Moyennes
- Comment déterminer $k$?
- k-Moyennes dans un espace non-euclidien
- Comparaison de k-Moyennes, GMM et DBSCAN

### Librairie scikit-learn
La librairie scikit-learn nous sera très utile puisqu'elle est dédiée à l'apprentissage machine. C'est une librarie libre qui contient de nombreux modèles utiles pour la classification, la régression, le partitionnement de données (clustering), la réduction de dimensionalité, etc. Au lieu de réinventer la roue et de réimplémenter à chaque fois un algorithme classique d'apprentissage machine, il est fortement recommandé d'utiliser cette librairie. Scikit-learn a été conçu de manière à être utilisable avec d'autres librairies scientifiques comme NumPy et SciPy. 

NumPy est une librairie très utile pour la manipulation de matrices. SciPy, qui se base sur NumPy, comprend un large ensemble de modules portant sur l'optimisation, le traitement d'image, l'algèbre linéaire, etc.

> **Ressources:**

> La documentation de la librairie est complète et elle contient de nombreux exemples, je vous invite à la consulter:
http://scikit-learn.org/stable/

> Pour un tutoriel de base, consultez:
http://scikit-learn.org/stable/tutorial/basic/tutorial.html

**tl;dr** Scikit-learn est votre ami! Lorsque vous le pouvez, utilisez-le au lieu de tout réimplémenter par vous-même!

<!-- #region -->
### Partitionnement des données
Le partitionnement des données consiste à partitionner l'ensemble de nos données en différents groupes, et ce, sur la base d'une distance. On souhaite minimiser la distance intra-groupe et maximiser la distance inter-groupe. Le but est d'ainsi découvrir des structures sous-jacentes à nos données. Cependant, comparativement au problème de classification, le problème du partitionnement des données est mal défini (qu'est-ce qu'un groupe?). Dans cette section, nous verrons tout d'abord des cas où certains algorithmes présentés en cours obtiendront de mauvais résultats. Par la suite, nous verrons une mesure nous permettant de trouver un nombre de centroïdes $k$ approprié pour la méthode des k-Moyennes. Finalement, nous verrons une version modifiée de k-Moyennes basé sur un exemple prototype plutôt que sur un centroïde.

## k-Moyennes

Avant de plonger dans cette section, voici un rappel de l'algorithme de k-Moyennes:
- Choisir $k$ centroïdes aléatoirement, ${m_1, ..., m_k}$
- Répéter si la condition d'arrêt n'est pas satisfaite
    - Assigner chaque point au centroïde le plus proche, $G_i = \{x \vert \forall j \neq i, D(x, m_i) \leq D(x, m_j)\}$
    - Recalculer le centroïde pour chaque groupe, $m_i = \frac{1}{|G_i|} \sum_{x \in G_i} x$ 

Vous pouvez remarquer qu'il y a principalement deux étapes qui sont semblables à l'algorithme espérance-maximisation (EM). Il y a d'abord une étape E où tous les points sont assignés à un groupe. Ensuite, il y a une étape M où l'on détermine les nouveaux centroïdes. Ces deux étapes sont répétées jusqu'à une certaine condition d'arrêt, par exemple un nombre d'itérations prédéterminé.

Pour voir k-Moyennes en action et l'essayez sur différents jeux de données, je vous invite à essayer par vous-même l'algorithme: https://www.naftaliharris.com/blog/visualizing-k-means-clustering/. Prenez le temps de bien voir les deux grandes étapes de l'algorithme.

> **Note:** par simplicité, nous avons choisi les centroïdes initiaux aléatoirement. Sachez qu'il existe d'autres méthodes qui permettent à k-moyennes de converger plus rapidement. Dans scikit-learn, il est possible de décider comment cette sélection se fait en changeant la valeur du paramètre `init` (voir la documentation de [`kMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)). Par exemple, il est possible d'utiliser la technique d'initialisation `k-means++` qui choisira des centroïdes initiaux distants afin de converger plus rapidement. En général, si vous souhaitez utiliser une version modifiée d'un algorithme existant, prenez le temps de regarder la documentation de scikit-learn. Il est probable qu'il soit possible de simplement changer un paramètre pour répondre à vos besoins!


Comme toute méthode, le k-Moyennes a plusieurs *a priori* ("assumptions"). Le k-Moyennes assume que les groupes sont sphériques, ont des tailles semblables et on un nombre équivalent d'exemples. Lorsque ces conditions ne sont pas respectées, il est très probable que la méthode trouve un minimum local et retourne un "mauvais" résultat. Pour illustrer ce phénomène, testons k-Moyennes sur des données ne respectant pas ces *a priopri*.
<!-- #endregion -->

Dans cette section, nous utiliserons le module `sklearn.datasets` qui permet de générer des données synthétiques. Nous générerons 1000 points (en deux dimensions) étant échantillonés à partir de trois gaussiennes isotropiques ayant une variance égale grâce à la méthode `make_blobs`.

```python
%matplotlib inline
# la ligne ci-haut permet d'afficher des graphiques dans le jupyter notebook sans utiliser la méthode show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1000
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=3)

plt.style.use('ggplot')

# on affiche nos données. 
# Comme premier argument on doit fournir les valeur en abcisse et 
# comme deuxième argument les valeurs de l'ordonnée
plt.scatter(x=X[:, 0], y=X[:, 1]);
```

Contrairement à l'exemple de la dernière démonstration, il n'y a pas d'étiquettes (*labels*) pour chacun des points. Nous sommes dans un contexte non-supervisé. Appliquons la méthode k-Moyennes à nos données et visualisons les groupes découverts.

```python
y_pred = KMeans(n_clusters=3).fit_predict(X)
print(y_pred[:20])
plt.scatter(X[:, 0], X[:, 1], c=y_pred);
```

Vous pouvez constater qu'il est très facile d'appliquer `kMeans`. Nous devons seulement spécifier le `n_clusters` qui correspond à $k$ et passer les données à la méthode `fit_predict`. Pour bien comprendre ce que nous retourne la méthode `fit_predict`, nous avons affiché ses 20 premières valeurs. Pour chaque point, `fit_predict` nous a retourné un nombre qui correspond au groupe déterminé par k-moyennes. Pour visualiser le résultat, nous pouvons fournir cette prédiction comme valeur pour le paramètre de couleur `c` à `plt.scatter`.


Comme ces données respectent suffisament bien les *a priori* de k-Moyennes, le résultat obtenu respecte notre intuition. Maintenant, appliquons k-Moyennes à des données qui ne sont pas tirés d'une gaussienne isotropique.

```python
# ici, nous appliquons une transformation à nos données
transformation = [[0.61, -0.64], [-0.41, 0.85]]
X_aniso = np.dot(X, transformation)
 
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred);
```

Comme les différents groupes ne sont pas sphériques, k-moyenne échoue lamentablement... Cette fois-ci utilisons des données provenant de gaussiennes isotropiques ayant des variances très différentes.

```python
X_varied, y_varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred);
```

Utilisons des données ayant des groupes de tailles différentes.

```python
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,random_state=random_state).fit_predict(X_filtered)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred);
```

Pour ce graphique, plusieurs seront d'accord avec les groupes trouvés. Cependant, on pourrait aussi argumenter que les points à droite (jaunes) ne forment pas véritablement un groupe, mais sont plutôt des valeurs aberrantes. Ceci illustre bien le caractère mal défini de la tâche de partitionnement.

Finalement, appliquons k-Moyennes aux données originales, mais avec un $k$ ne correspondant au nombre de gaussiennes utilisées par `make_blob`.

```python
y_pred = KMeans(n_clusters=4).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred);
```

Berf, vous pouvez constater que si les *a priori* de la méthode k-moyennes ne sont pas respectés, alors les résultats peuvent être douteux... 


## Hyperparamètre: comment déterminer $k$?

Jusqu'à maintenant nous avons utilisé des données seulement en deux dimensions. Il était très facile de proposer un nombre de groupe ($k$) pertinent pour k-Moyennes. Cependant, en pratique, la plupart des jeux de données ont un grand nombre de dimensions et il ne sera pas possible de déterminer visuellement le $k$ approprié. Alors comment faire?

Il existe plusieurs mesures permettant d'évaluer la cohésion intra-groupe et la séparation inter-groupe. Par exemple, le score silhouette (<https://en.wikipedia.org/wiki/Silhouette_(clustering)>) est un score associé à chaque point qui a une valeur entre -1 et 1. Pour évaluer le partitionnement, nous utiliserons la moyenne du score silhouette pour l'ensemble de nos données. Une valeur élevée signifie que le partitionnement est adéquat, inversement une valeur basse signifie que le $k$ choisi est inadéquat.

Pour un point $i$, son score silhouette $s$ est donné par:
$$ s(i) = \frac{b(i) - a(i)}{\max{\{a(i), b(i)}\}} $$
où $a(i)$ est la distance moyenne entre le point $i$ et l'ensemble des points faisant partie du même groupe, $b(i)$  est la distance moyenne entre le point $i$ et l'ensemble de tous les points ne faisant pas partie du même groupe. En général, un point bien assigné aura un $a(i)$ peu élevé par rapport à $b(i)$.


Pour déterminer le $k$ approprié, nous appliquerons la méthode k-Moyennes avec différents $k$ et nous calculerons le score silhouette correspondant pour chacun. Nous choisirons le $k$ associé au plus haut score silhouette. Le score silhouette est implémenté dans scikit-learn, inutile de l'implémenter par nous-même!

```python
from sklearn.metrics import silhouette_score

scores = []
k_range = range(2,15)
for k in k_range:
    y_pred = KMeans(n_clusters=k).fit_predict(X)
    scores.append(silhouette_score(X, y_pred))
```

```python
plt.plot(k_range, scores)
plt.xlabel('k')
plt.ylabel('Score silhouette')
plt.title('Score silhouette en fonction de k');
```

Pour chaque point du graphique, nous avons appliqué k-Moyennes avec un $k$ différent. Pour chacun des partitionnements, le score silhouette moyen a été calculé. Comme attendu, le $k$ ayant un score silhouette le plus élevé est 3, soit le nombre de gaussiennes provenant de `make_blob`. Avec ces données, le résultat est évident. Cependant, avec des données "naturelles", il est souvent plus ardu de déterminer le $k$ adéquat. Essayons la même méthode avec un nouveau jeu de données "mystère"...

```python
X = np.loadtxt('silhouette.csv')

scores = []
k_range = range(5,20)
for k in k_range:
    y_pred = KMeans(n_clusters=k).fit_predict(X)
    scores.append(silhouette_score(X, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('k')
plt.ylabel('Score silhouette')
plt.title('Score silhouette en fonction de k');
```

Quel est la bonne valeur de $k$? Est-ce 18 où le maximum est atteint? Ou bien est-ce 10 qui a une valeur semblable à 18, mais qui a nombre de groupe inférieur? Il n'y a pas de réponse définitive. Cependant, le jeu de données mystère était en fait MNIST (les images de chiffres de 0 à 9 écrits à la main) ayant subi une réduction de dimensionnalité. Ce qui nous laisse croire que 10 est la bonne réponse (car il y a précisément 10 classes dans MNIST). Cependant, comme certains chiffres ont des graphies différentes (ex. le chiffre 7), il est possible que notre méthode de partitionnement prenne en compte ces graphies et les considère comme des groupes.

Voici le partitionnement obtenu:

```python
y_pred = KMeans(n_clusters=10).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred);
```

En général, il faut être prudent avec cette méthode pour trouver le $k$. Par exemple, si $k = |D|$ (où $D$ est l'ensemble des données), on obtiendra alors un score très élevé, cependant les groupes découverts ne seront pas du tout pertinents... Chaque donnée aura son propre groupe!

Il existe naturellement d'autres manières de déterminer le $k$ adéquat. Par exemple, certaines méthodes pénalisent pour des valeurs de $k$ trop élevées. Si le sujet vous intéresse, je vous invite à consuler : https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set


## k-Moyennes dans un espace non-euclidien

Supposons que nous avons des données dans un espace non-euclidien pour lequel nous avons tout de même une notion de  distance. Dans ce cas, il peut être pertinent d'utiliser une variante de la méthode k-Moyennes nommé k-Médoïde. Au lieu de trouver un centroïde pour représenter chacun des groupes, un exemple sera utilisé pour représenter chacun des groupes.

Afin d'illustrer k-Médoïde, prenons un exemple - un peu artificiel - constitué d'un petit jeu de données de mots. Supposons que ces mots sont écrits par des utilisateurs et permettent de déterminer la langue de l'utilisateur. Cependant, les utilisateurs font beaucoup d'erreurs de frappe. Il n'est donc pas possible de simplement utiliser un dictionnaire (ex. bonjour => français, hello => anglais). Dans cette exemple, un centroïde (i.e., la moyenne du groupe) n'a pas de sens. En effet, quel est le point milieu entre "bonjour" et "hello"? La question est absurde...

```python
words = ['bonjour', 'bonkour', 'bonjoru', 'bonjjour', 'bionjour', 'hello', 'helo', 'hrllo', 'ello', 'yello',
         'helllo','konnichiwa', 'konichiwa', 'konnchiwa', 'konnichioua', 'connichiwa']
```

Mais comment évaluer la similarité de deux mots? Est-ce que "hello" est plus proche de "pomme" que de "pelle"? Une distance intéressante est la distance de Levenshtein (https://fr.wikipedia.org/wiki/Distance_de_Levenshtein), qui est aussi nommée distance d'édition. Cette distance entre deux chaînes de caractères est égale au nombre minimal de suppression, ajout et remplacement nécessaire pour que les deux chaînes soient la même. Par exemple, entre les mots "pommes" et "assomme", il y a une distance de Levenshtein de 4. En effet, pour transformer "assomme" en "pommes" il faut tout d'abord supprimer les deux caractères "as" (2 modifications), remplacer le "s" pour un "p" (1), puis ajouter un "s" (1). Le tout pour un total de 4 modifications.

Ici, nous utiliserons un module python permettant de calculer la distance de Levenshtein. Pour vérifier notre exemple:

```python
import Levenshtein
Levenshtein.distance('pommes', 'assomme')
```

> **Note:** Il est fort probable que le module Levenshtein ne soit pas installé sur votre ordinateur. Si c'est le cas, vous pouvez utiliser la commande `pip install python-Levenshtein` dans votre terminal.


Maintenant que nous avons une distance pour comparer les mots, nous pourrons appliquer la méthode k-Médoïde. Contrairement à k-Moyennes, l'implémentation de k-Médoïde nécessite qu'on lui fournisse la matrice de distance de nos données. À chaque position $(i,j)$ de cette matrice correspond la distance entre l'exemple $i$ et l'exemple $j$.

```python
D = np.zeros((len(words), len(words)))
for i in range(len(words)):
    for j in range(len(words)):
        D[i,j] = Levenshtein.distance(words[i], words[j])
```

> **Note:** Malheureusement, il n'y a pas d'implémentation de k-Médoïde dans scikit-learn. L'implémentation utilisée ici vient de https://pypi.org/project/pyclustering/0.7.0/. Pour que l'exemple fonctionne, assurez-vous d'avoir installé cette librairie.

```python
from pyclustering.cluster.kmedoids import kmedoids

initial_medoids = [0,1,2]

# ici, nous donnons directement notre matrice de distance
# il faut spécifier data_type='distance_matrix'
kmedoids_instance = kmedoids(D, initial_medoids, data_type='distance_matrix')

kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()
```

```python
print('Medoides:')
for i in medoids:
    print(words[i])
    
print('\nAssignation aux groupes:')
for label, cluster in enumerate(clusters):
    for i in cluster:
        print('label {0}:　{1}'.format(label, words[i]))
```

On peut tout d'abord constater que la méthode nous retourne les valeurs des médoïdes. Ces valeurs correspondent justement aux mots ayant une bonne orthographie! Dans un second temps, on peut voir que les différents groupes trouvés correspondent bien à notre intuition.


## Comparaison de k-Moyennes à d'autres modèles de partitionnement

Nous avons vu plusieurs exemples où la méthode k-Moyennes échoue. Habituellement, cela survient lorsque les données ne respectent pas nos *a priori*. Dans cette section, nous verrons comment le modèle de mélange de Gaussiennes et DBSCAN performent sur les mêmes données. Et, finalement, nous verrons les limites de ces deux modèles.


### Modèle de mélange de Gaussiennes 

Nous avons vu que k-Moyennes peut avoir de sérieux problèmes lorsque les données ne sont pas réparties en sphère avec une variance égale. En fait, ce modèle peut être vu comme un cas particulier du modèle de mélange de Gaussiennes (GMM) où les gaussiennes sont isotropiques (la variance de chaque dimension est la même et les dimensions sont indépendantes). En général, le GMM peut partitionner convenablement des données provenant de normales non-isotropiques.

```python
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=n_samples, random_state=random_state, centers=3)
transformation = [[0.61, -0.64], [-0.41, 0.85]]
X_aniso = np.dot(X, transformation)
 
estimator = GaussianMixture(n_components=3, covariance_type='full', max_iter=20, random_state=0)
estimator.fit(X_aniso)
y_pred = estimator.predict(X_aniso)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred);
```

L'utilisation de `GaussianMixture` est légèrement différente de `kMeans`. Il faut tout d'abord initialiser `GaussianMixture`, ensuite utiliser sa méthode `fit` et finalement `predict`. La différence provient du fait que `GaussianMixture` ne fait pas partie du module `sklearn.cluster`. En effet, comme vous le verrez cette méthode peut aussi être utilisé pour faire de la classification.


Maintenant, utilisons un GMM sur des données qui ne proviennent pas d'une distribution normale. En l'occurence, nous utiliserons `make_moons`.

```python
from sklearn.datasets import make_moons

n_samples = 1000
random_state = 170
X_moon, _ = make_moons(n_samples=n_samples, random_state=random_state, noise=0.1)

estimator = GaussianMixture(n_components=2, covariance_type='full', max_iter=20, random_state=0)
estimator.fit(X_moon)
y_pred = estimator.predict(X_moon)
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_pred);
```

### DBSCAN

Contrairement à k-Moyennes et au GMM, il n'est pas nécessaire de spécifier le nombre de groupe attendu pour DBSCAN. De plus, DBSCAN n'assume pas de forme particulière des groupes.

DBSCAN a tout de même deux hyperparamètres qui peuvent grandement influencer le résultat obtenu: le rayon de voisinage et la densité critique.

Pour tester par vous-même: https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/

```python
from sklearn.cluster import DBSCAN

y_pred = DBSCAN().fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred);
```

L'utilisation de `DBSCAN` est presque identique à `kMeans`: il suffit d'utiliser la méthode `fit_predict`. Nous n'avons pas spécifier de valeur pour les hyperparamètres. Ces valeurs sont par défaut `eps=0.5, min_samples=5`, c'est-à-dire un rayon de 0.5 et une densité critique.

Sur ces données, le résultat est semblable à k-Moyennes. Cependant, DBSCAN a -en plus- identifié des points aberrants. 

```python
y_pred = DBSCAN(eps=0.1, min_samples=5).fit_predict(X_moon)
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_pred);
```

Comme DBSCAN n'a pas d'*a priori* concernant la forme de la distribution des données, le jeu de données `make_moons` ne constitue pas un problème. Appliquons maintenant DBSCAN aux mêmes données où k-Moyennes échouait:

```python
plt.figure(figsize=(18, 6))

transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = DBSCAN(eps=0.3, min_samples=5).fit_predict(X_aniso)

plt.subplot(131)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Groupe non-spherique")

X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = DBSCAN().fit_predict(X_varied)

plt.subplot(132)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Variance inegale")

X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = DBSCAN().fit_predict(X_filtered)

plt.subplot(133)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Taille des groupes debalancee");
```

Ici, nous avons utilisé `subplot` qui peut être très pratique pour afficher plusieurs graphiques en même temps.


Pour terminer, appliquons DBSCAN aux gaussiennes non-isotropiques, mais avec des hyperparamètres différents...

```python
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_aniso)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred);
```

Ici, le rayon de voisinage (`eps` dans le code) est trop grand et toutes les données sont groupées dans un seul grand groupe.


## Conclusion

Pour résumer cette démonstration, nous avons vu que chaque méthode a ses points faibles et ses points forts. Il est important d'en tenir compte lorsqu'on les utilise! Nous avons vu une version modifiée de k-Moyennes permettant de traiter des données qui ne sont pas dans un espace euclidien. Nous avons aussi vu une méthode qui permet de trouver le meilleur hyperparamètre $k$ pour notre méthode k-Moyennes. De plus, tout au long de la démonstration, nous nous sommes familiarisé avec la super librairie scikit-learn!


(Exemples inspirés de http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html)


# Exercices
1. Comme l'objectif de k-Moyennes est non-convexe, cet algorithme retourne souvent des minimums locaux: c'est-à-dire des partitionnements qui ne sont pas optimaux. Or, en pratique, ce n'est pas un problème majeur, car l'algorithme est appliqué plusieurs fois avec des _seeds_ différents. Pour mettre en évidence le phénomène des minimums locaux, utilisez les données `X` ici bas, fixez l'argument `init` de `kMeans` à `'random'` et changez un argument de `kMeans` afin d'utiliser un seul _seed_ (Référez-vous à la documentation). Appliquez l'algorithme plusieurs fois et affichez les différents partitionnements trouvés par `kMeans`. Constatez comment les résultats peuvent grandement varier...

```python
n_samples = 100
random_state = 1
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=3)
plt.scatter(x=X[:, 0], y=X[:, 1]);
```

### Solution 1.

```python
y_pred = KMeans(n_clusters=3, init="random", n_init=1, random_state=5).fit_predict(X)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y_pred);
```

Ce premier exercice simple avait deux objectifs. D'une part, vous habituer à consultez la documentation de scikit-learn qui vous sera très utile pour le reste du cours. D'autre part, montrer le phénomène de convergence à un minimum local. On peut voir que bien que le problème de partitionement semble facile (il y a trois groupes bien distincts), certaines initialisations mènent à de médiocres résultats. Dans les prochains cours, vous verrez d'autres méthodes qui peuvent converger dans des minimums locaux. Gardez en tête que différentes initialisations peuvent donner des résultats très différents!

```python

```

2. Trouvez des valeurs de `epsilon` et `min_samples` pour l'algorithme DBSCAN qui retourne un "bon partitionnment" des données `X` ici bas (données tirées d'un exemple de scikit-learn), c'est-à-dire 4 clusters. Vous pouvez tester par essai-erreur ou bien utilisez la méthode `OPTICS`...

```python
np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))
plt.scatter(X[:,0], X[:,1]);
```

### Solution 2.


La méthode OPTICS est semblable à DBSCAN, mais elle permet de trouver une bonne valeur de epsilon. Pour commencer, essayons d'appliquer la méthode directement.

```python
from sklearn.cluster import OPTICS

clust = OPTICS(min_samples=50, max_eps=10)
y_pred = clust.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=8);
```

Le résultat n'est pas convaincant. On peut bien voir la source du problème: la densité n'est pas du tout uniforme. À l'intérieur d'un des clusters, il y a deux autres clusters très denses. OPTICS nous permet de voir les différents partitionnements en fonction de la valeur de epsilon. Réappliquons OPTICS et trouvons une bonne valeur de epsilon.

```python
from sklearn.cluster import OPTICS

clust = OPTICS(min_samples=50, max_eps=10)
clust.fit(X)

space = np.arange(X.shape[0])
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    plt.plot(Xk, Rk, color, alpha=0.3)
plt.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3);
```

Pour comprendre ce graphique, voici quelques brèves explications de OPTICS. OPTICS a deux hyperparamètres: `min_samples` qui correspond à la densité critique et `max_eps` qui correspond au rayon maximal qui sera considéré. Un point est considéré comme un point milieu si sa `core distance` est plus petite que `max_eps`. La `core distance` d'un certain point est le rayon minimal nécessaire pour que ce point ait au moins `min_samples` voisins. La `reachability distance` d'un point $p$ à un point milieu $q$ est le rayon minimal pour inclure $p$ comme voisin de $q$ et pour $q$ ait au moins `min_samples` voisins. En d'autres termes, c'est $max(coredistance(q), distance(p,q))$. Maintenant que ces concepts ont été expliqués, analysons le graphique. L'axe des x est l'ordre des points dans lequel l'algorithme s'est propagé et l'axe des y est la `reachability distance` de chaque point. On peut y voir plusieurs "vallées". Ces vallées correspondent à des clusters. En effet, lorsque la valeur de `reachability distance` est faible pour plusieurs points, cela signifie que les points sont tous près les un des autres et forme un cluster. À l'aide du graphique, on peut voir qu'en prenant un epsilon de 2 (ou environ entre 1.5 et 2), on pourra trouver les clusters qui nous intéressent. Si on choisit un epsilon trop faible, par exemple de 1, tout le groupe jaune sera considéré comme du bruit (car ce clusters n'est pas très dense).

```python
from sklearn.cluster import DBSCAN

y_pred = DBSCAN(eps=2, min_samples=50).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred);
```

On peut constater qu'un epsilon de 2 donne effectivement un bon résultat. L'avantage de OPTICS par rapport à la méthode essai-erreur, c'est qu'il a suffit d'appliquer la méthode OPTICS qu'une seule fois pour déterminer la valeur de epsilon. Naturellement, cette méthode ne résout pas tous les problèmes: il faut tout de même fournir une densité critique et un epsilon maximal. 
> __NOTE:__ Comme cette méthode n'a pas été présentée en cours et que mes explications se veulent brèves, elle ne sera pas matière à examen.

```python

```

3. Déterminez le nombre de regroupement dans un vrai jeu de données à partir de l'algorithme k-moyennes et du score silhouette. Rendez-vous sur la page suivante: [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29), cliquez sur `Data Folder`, puis sur `breast-cancer-wisconsin.data`. Ce jeu de données contient des résultats de biopsie de tumeurs du sein. Utilisez les lignes suivantes pour obtenir les données (assurez-vous d'avoir téléchargé le jeu de données et de le mettre dans le même dossier que votre notebook).

```python
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data', sep=',', header=None)
df = df.drop(columns = [0, 6])
```

```python
df.head()
```

```python
X = df.values
```

### Solution 3.

```python
scores = []
k_range = range(2,15)
for k in k_range:
    y_pred = KMeans(n_clusters=k).fit_predict(X)
    scores.append(silhouette_score(X, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('k')
plt.ylabel('Score silhouette')
plt.title('Score silhouette en fonction de k');
```

Selon le graphique, le bon nombre de partitions selon le $k$ semble être 2. Cependant, selon la littérature médicale, on parle souvent de 4 types de cancer du sein. Certes, notre graphique semble, dans une faible mesure, confirmer cela: on voit un plateau dans le score pour $k = [3,5]$. Mais comment expliquez que ce ne soit pas plus évident? Il y a certainement plusieurs explications possibles. Entre autre, il se peut que ce soit les données que nous utilisons. Par exemple, les données peuvent être trop bruitées ou bien le type de données (biopsie) ne donne pas assez d'informations pour départager les types de cancer. Bref, cet exercice vous montre que, bien que les scores d'indice de partitionnement soient utiles, ils sont assez limités et il ne faut pas s'y fier aveuglément. De plus, contrairement aux exemples synthétiques, ce n'est pas évident de tirer une conclusion. Entre autre, il n'est pas possible de directement visualiser les résultats puisqu'ils sont en 9 dimensions.

```python

```
