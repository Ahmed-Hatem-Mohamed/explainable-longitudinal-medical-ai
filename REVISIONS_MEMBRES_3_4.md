# Révisions techniques à intégrer — Membres 3 et 4

Cette fiche complète le plan NACC V1. Elle précise les points à verrouiller avant l'entraînement et l'évaluation du modèle longitudinal multimodal.

**Document de référence :** [NACC Project Technical Specification and Gantt](NACC_Project_Technical_Specification_and_Gantt.pdf). La spécification initiale ([Technical Specification V1](Technical_Specification_V1.pdf)) contient encore des choix plus généraux, notamment ADNI et l'imagerie 3D ; pour cette V1, le présent document suit le choix NACC et les mesures régionales d'IRM.

## Décisions communes aux membres 3 et 4

- Utiliser les mêmes cohortes, labels, règles d'exclusion et partitions patient pour tous les modèles comparés. Un patient ne doit apparaître que dans une seule partition.
- Calculer les imputations, critères de sélection des variables, moyennes/écarts-types et autres paramètres de prétraitement sur l'entraînement seulement, puis appliquer ces paramètres à la validation et au test.
- Verrouiller la définition du label : progression de MCI vers démence Alzheimer dans les 730 jours après la visite d'ancrage, avec un suivi suffisant pour déterminer les négatifs.
- Documenter les dates et les informations disponibles à chaque prédiction pour prévenir toute fuite depuis le futur. Aucun diagnostic postérieur à l'ancrage ni donnée calculée à partir du suivi futur ne peut entrer dans les variables prédictives.
- Traiter les scores et explications comme des descriptions du comportement du modèle, pas comme des preuves de causalité clinique.

## Membre 3 — Modèle longitudinal multimodal

### Architecture et entrées

- Conserver l'architecture V1 proposée : encodeur MRI régional, encodeur clinique, fusion par visite, GRU et tête de prédiction. Elle est adaptée à des variables MRI régionales tabulaires ; elle ne constitue pas un modèle d'imagerie 3D brute.
- Donner à l'encodeur MRI les valeurs et les indicateurs de valeurs manquantes (`2M` dimensions) et à l'encodeur clinique les six valeurs et leurs six indicateurs (`12` dimensions), conformément au contrat `PatientSequence`.
- Vérifier que les variables MRI conservées sont numériques et anatomiquement interprétables. Exclure les identifiants, dates, variables administratives/QC et informations de scanner qui ne sont pas destinées au modèle.
- Définir l'encodage du temps écoulé : unité, transformation éventuelle et normalisation ajustée sur l'entraînement. Fixer explicitement la valeur de la première visite et son masque.

### Cible et trajectoire de risque

Le plan définit le label à partir d'une visite d'ancrage et de son suivi ultérieur de 730 jours. Il faut distinguer deux usages possibles des prédictions par préfixe :

1. **Option recommandée pour la V1 :** entraîner la tête de prédiction sur la dernière visite valide, correspondant à l'ancrage. Les sorties calculées sur les préfixes antérieurs peuvent illustrer l'évolution des scores quand le modèle reçoit plus d'historique, mais ne doivent pas être présentées comme des risques prospectifs calibrés à la date de chaque préfixe.
2. **Option à retenir uniquement si le besoin est confirmé :** prédire un événement à horizon de 24 mois à partir de chaque visite. Il faut alors construire un label et une fenêtre de suivi propres à chaque date de prédiction, exclure les préfixes sans suivi suffisant et réévaluer la disponibilité des données.

Ne pas appliquer automatiquement le même label d'ancrage à chaque préfixe tout en affirmant que chacun prédit le risque des 24 mois qui suivent sa propre date.

### Séquences, temps et entraînement

- Implémenter le padding de façon à ce qu'une visite ajoutée uniquement pour compléter un batch n'influence ni l'état final de la GRU, ni la perte, ni les métriques. Utiliser un mécanisme de séquences empaquetées ou un masquage correctement appliqué à chaque étape.
- Si la perte est calculée à plusieurs étapes, ne la calculer que sur les étapes valides et associer à chaque étape le label qui correspond à la définition de prédiction retenue.
- Garder les pondérations de classes et seuils sous contrôle : `pos_weight` est déterminé à partir de l'entraînement ; tout seuil de décision est choisi sur la validation et gelé avant l'évaluation finale sur le test.
- Fixer les graines et enregistrer configuration, prétraitements, versions, historique d'entraînement, checkpoint et contrat des entrées/sorties afin de pouvoir reproduire une inférence.

### Tests et critères de fin pour le membre 3

Le workstream est prêt lorsque :

- une vraie séquence NACC conforme au contrat passe de bout en bout jusqu'à une probabilité et, selon la définition retenue, aux scores de préfixe ;
- les formes, masques de valeurs manquantes, masques de séquence et intervalles de temps sont validés, y compris pour des séquences de longueurs différentes ;
- ajouter du padding ne change pas les sorties valides ; l'inférence d'un patient seul correspond à son résultat dans un batch ;
- les tests confirment que les prétraitements sont ajustés sur l'entraînement seulement et qu'aucune donnée postérieure à la date de prédiction n'est utilisée ;
- le contrat de sortie précise si chaque score est une prédiction à l'ancrage ou un risque recalculé à chaque visite.

## Membre 4 — Évaluation et explicabilité

### Comparaisons nécessaires

Comparer les modèles sur les mêmes patients, labels, partitions et protocole de prétraitement. Pour isoler l'apport des modalités et de l'historique, inclure au minimum :

| Modèle | Modalités | Historique | Question évaluée |
|---|---|---|---|
| Référence simple (prévalence ou régression logistique) | Variables cliniques, dernière visite | Non longitudinal | Quel niveau obtient une référence simple ? |
| Clinique seule | Clinique | Dernière visite seulement | Que prédisent les variables cliniques sans historique ? |
| MRI seule | MRI régionale | Dernière visite seulement | Que prédisent les mesures MRI sans clinique ni historique ? |
| Multimodal non longitudinal | Clinique + MRI | Dernière visite seulement | Quel est l'apport de la fusion sans séquence ? |
| Clinique longitudinale | Clinique | Plusieurs visites | Quel est l'apport de l'historique clinique ? |
| MRI longitudinale | MRI régionale | Plusieurs visites | Quel est l'apport de l'historique MRI ? |
| Multimodal longitudinal V1 | Clinique + MRI | Plusieurs visites, GRU | Le modèle complet améliore-t-il les références ? |

Les modèles « dernière visite seulement » sont non longitudinaux, même s'ils utilisent la même visite d'ancrage. Si le temps ou les visites antérieures sont nécessaires à la définition du label/cohorte, ils restent identiques entre les bras de comparaison ; seules les entrées du modèle changent. Garder également la régression logistique sur la dernière visite prévue dans le plan, mais préciser ses variables d'entrée et son traitement des valeurs manquantes.

### Protocole de performance

- Rapporter AUROC et AUPRC, ainsi que sensibilité, spécificité, balanced accuracy, F1, Brier score et courbe de calibration. Indiquer la prévalence et les effectifs de chaque partition.
- Choisir le seuil uniquement sur la validation selon la règle décidée à l'avance ; rapporter ensuite les métriques de test sans ajustement supplémentaire.
- Quand les effectifs le permettent, donner des intervalles de confiance calculés par rééchantillonnage au niveau patient, pas au niveau visite.
- Si une calibration est appliquée, l'ajuster sur la validation uniquement et décrire la méthode. Ne pas calibrer ni sélectionner de modèle à partir du test.
- Présenter les résultats des ablations de modalités comme des expériences distinctes. Les entrées retirées doivent être masquées ou traitées selon une règle explicite ; une absence artificielle de modalité ne doit pas être confondue avec le mécanisme réel de données manquantes.

### Explications : portée et stabilité

- Pour les mesures MRI régionales de cette V1, nommer les résultats « attribution aux variables/régions anatomiques mesurées ». Ne pas promettre de cartes d'importance voxel par voxel, qui nécessiteraient des images et une méthode spatiale adaptées.
- Pour Integrated Gradients, définir et documenter une référence (baseline) pertinente dans l'espace prétraité, par exemple le profil moyen de l'entraînement après standardisation. Vérifier la convergence/complétude des attributions et préciser comment elles sont agrégées par visite, variable et modalité.
- Évaluer la stabilité des explications sur des perturbations faibles et réalistes des entrées, plusieurs initialisations ou réentraînements lorsque le budget le permet, et plusieurs références plausibles pour Integrated Gradients. Comparer les classements des variables (par exemple corrélation de rang ou recouvrement top-k) et indiquer les cas où les explications changent fortement.
- Évaluer la fidélité au comportement du modèle par retrait contrôlé de visites ou groupes de variables et par comparaison avec les attributions annoncées. Une attribution élevée devrait, dans les limites du test, correspondre à une variation mesurable de la sortie lorsque l'entrée concernée est retirée ou masquée.
- Pour l'importance temporelle, comparer l'inférence complète à une ablation « une visite retirée à la fois », en conservant des règles cohérentes de longueur, de masque et de temps. Rapporter les cas où une séquence modifiée devient hors distribution.
- Décrire les explications comme des indicateurs de sensibilité du modèle ; elles ne démontrent ni mécanisme biologique ni relation causale.

### Tests et critères de fin pour le membre 4

Le workstream est prêt lorsque :

- les résultats comprennent les comparaisons unimodales, non longitudinales et longitudinales du tableau, évaluées sur une partition patient identique ;
- le seuil et toute calibration sont déterminés sans consulter le test ; les métriques, effectifs, prévalence et intervalles de confiance sont consignés ;
- `explain(patient)` renvoie des attributions cliniques et MRI régionales, une importance temporelle et une attribution par modalité, avec référence et méthode documentées ;
- un rapport de stabilité et de fidélité indique la méthode, les perturbations, la mesure de comparaison, les limites et les cas instables ;
- les visualisations distinguent clairement contribution du modèle, ablation expérimentale et interprétation clinique.

## Calendrier

Le Gantt du document de référence prévoit une livraison le 15 octobre 2026, date passée. Avant d'attribuer de nouveaux délais aux membres 3 et 4, relever l'état réel des données, des labels, du modèle, des baselines et des tests, puis établir un calendrier révisé à partir de ces dépendances. Cette fiche ne suppose pas que les jalons antérieurs ont été réalisés.
