# Air Paradis : construire, comparer et déployer un modèle de sentiment dans une logique MLOps

L’analyse de sentiment sur des tweets est un exercice classique en data science. Mais transformer ce type de modèle en un service réellement exploitable est une autre histoire. Dans le projet Air Paradis, l’objectif n’était pas seulement d’obtenir un bon score sur un jeu de test, mais de construire un système complet : comparer plusieurs approches, suivre les expérimentations, déployer une API sur Hugging Face et mettre en place un mécanisme de monitoring capable de détecter un comportement anormal.

Autrement dit, passer d’un modèle “dans un notebook” à un système vivant.



## Comparer trois approches pour comprendre les compromis

### Le modèle sur mesure simple

La première étape a consisté à établir une base solide avec un modèle classique : nettoyage des tweets, vectorisation TF-IDF, puis régression logistique.

Cette approche présente plusieurs avantages. Elle est rapide à entraîner, rapide à prédire et simple à déployer. Elle constitue une référence claire pour mesurer les gains apportés par des modèles plus complexes.

Cependant, ce type de modèle reste limité par la nature même de la représentation TF-IDF. Il capture la fréquence et l’importance statistique des mots, mais ne comprend pas réellement le contexte. Sur des tweets courts, ambigus ou ironiques, ces limites apparaissent rapidement.



### Le modèle sur mesure avancé

La deuxième approche conserve une architecture classique mais introduit davantage de rigueur dans le pipeline : prétraitement plus maîtrisé, optimisation des hyperparamètres et gestion plus fine des déséquilibres.

Cette version avancée améliore les performances de manière significative tout en restant légère. Elle offre un compromis intéressant entre qualité prédictive et simplicité opérationnelle.

Dans un contexte de mise en production, ce compromis est crucial. Un modèle légèrement plus performant mais beaucoup plus lourd à déployer peut s’avérer moins pertinent qu’un modèle stable, rapide et facilement maintenable.



### Le modèle avancé BERT

La troisième approche repose sur un modèle Transformer de type BERT. Contrairement aux méthodes précédentes, ce modèle comprend le contexte d’une phrase dans sa globalité.

Les performances obtenues sont supérieures, notamment sur les tweets ambigus ou contenant des subtilités linguistiques. Toutefois, cette performance a un coût : entraînement plus long, dépendances plus lourdes, latence plus élevée.

Dans le cadre de ce projet, BERT a permis d’évaluer le potentiel maximal atteignable, mais la question n’était pas uniquement “quel modèle est le plus performant ?” Elle était aussi “quel modèle est le plus adapté à un déploiement réel ?”



## Choix du modèle pour la mise en production

Après comparaison des expérimentations suivies dans MLflow, le modèle sur mesure avancé a été retenu pour la mise en production.

Il offre un excellent équilibre entre performance, stabilité et simplicité de déploiement. Dans une logique MLOps, ce type de décision est stratégique. L’objectif n’est pas d’obtenir la meilleure métrique possible à tout prix, mais de livrer un système fiable, reproductible et surveillable.



## Une démarche MLOps structurée

### Suivi des expérimentations avec MLflow

Chaque entraînement a été encapsulé dans un run MLflow. Les paramètres, métriques et modèles ont été enregistrés systématiquement.

Cette approche permet d’éviter les décisions arbitraires. Les modèles sont comparés objectivement. Les choix sont traçables. Il devient possible de revenir à une version antérieure si nécessaire.

MLflow joue ici un rôle central dans la reproductibilité et la gouvernance des modèles.



### Organisation du code et gestion de version

Le projet est structuré de manière claire : notebooks pour l’exploration, code API séparé pour le déploiement, et fichier de dépendances explicite.

Le versioning via GitHub garantit la traçabilité des évolutions, notamment l’ajout progressif du système de feedback et du mécanisme d’alerte.

Cette organisation permet de séparer clairement les phases d’expérimentation et de production.



### Tests et validation

Un système déployé doit être fiable. Des tests simples mais essentiels ont été mis en place pour vérifier :

* le bon fonctionnement des endpoints,
* la validité des entrées et sorties,
* la logique de déclenchement de l’alerte.

Cette étape est fondamentale pour éviter qu’une modification ultérieure casse silencieusement le service.



## Déploiement sur Hugging Face Spaces

Le modèle sélectionné est exposé via une API FastAPI déployée sur Hugging Face Spaces.

L’API propose trois endpoints principaux :

* un endpoint de santé pour vérifier l’état du service,
* un endpoint de prédiction pour obtenir le sentiment d’un tweet,
* un endpoint de feedback permettant d’indiquer si la prédiction était incorrecte.

Le déploiement sur Hugging Face offre un environnement simple et accessible, tout en permettant une véritable mise en situation de production.



## Mise en place d’un mécanisme de monitoring

Un modèle en production peut se dégrader. Les données peuvent évoluer. Le comportement des utilisateurs peut changer. Il est donc essentiel de surveiller le système.

Dans ce projet, un mécanisme simple mais efficace a été implémenté. À chaque feedback négatif, un compteur interne s’incrémente. Lorsque trois prédictions consécutives sont signalées comme incorrectes, une alerte est déclenchée.

Cette alerte se traduit par :

* un message explicite dans les logs du service,
* une levée d’erreur HTTP volontaire permettant de matérialiser l’événement.

Ce dispositif simule une détection de dérive ou d’anomalie et démontre la capacité du système à réagir à un comportement inattendu.



## Vers une amélioration continue

Le monitoring ne doit pas être considéré comme une simple alarme. Il constitue un point de départ pour une boucle d’amélioration continue.

Les feedbacks négatifs peuvent être analysés pour identifier des motifs récurrents. Ces cas peuvent enrichir le dataset d’entraînement. Une nouvelle version du modèle peut ensuite être entraînée, comparée via MLflow et redéployée si elle apporte une amélioration mesurable.

On entre alors dans un cycle vertueux : données, modèle, production, feedback, amélioration.

C’est précisément cette capacité d’adaptation qui distingue un projet académique d’un système réellement exploitable.


 Conclusion

Le projet Air Paradis illustre le passage d’une expérimentation de machine learning à une solution déployée et surveillée.

La comparaison des trois approches a permis de faire un choix rationnel basé sur un équilibre entre performance et exploitabilité. L’intégration de MLflow assure la traçabilité des décisions. Le déploiement sur Hugging Face Spaces démontre la capacité à industrialiser le modèle. Enfin, le mécanisme d’alerte montre que le système ne se contente pas de prédire, mais qu’il peut détecter un comportement anormal.

Ce projet met en lumière une réalité importante : en machine learning, le modèle n’est qu’une partie du travail. La robustesse, la surveillance et la capacité d’évolution sont tout aussi essentielles.

