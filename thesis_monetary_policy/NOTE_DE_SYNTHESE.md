# Note de synthèse

## Impact de la politique monétaire de la Fed et de la BCE sur les rendements financiers

Sarah Madelyne BIZA · Master 1 Économétrie et Statistique, parcours Économétrie Appliquée · IAE Nantes · 2025-2026

**Résumé.** Cette étude porte sur comment les décisions de la BCE et de la Fed influencent les marchés boursiers français et américains. À partir de trois indices majeurs (CAC 40, S&P 500, NASDAQ 100), de neuf secteurs et de quatre actions, elle montre que les rendements dépendent des taux d'intérêt et de l'incertitude. Les effets diffèrent selon les périodes et les régimes monétaires, et les estimations par MCO doivent se lire avec leurs limites.

**Mots-clés** : politique monétaire, BCE, Fed, marché actions, taux d'intérêt, MCO

## I. Problématique et cadre de l'étude

Les décisions des banques centrales influencent les conditions financières mondiales. La Réserve fédérale américaine (Fed) et la Banque centrale européenne (BCE) sont particulièrement observées car leurs annonces modifient immédiatement le rendement des actions mais aussi la volatilité. Ce fait n'est pas nouveau mais il s'est accentué dans un contexte marqué par des chocs successifs (crise financière de 2008, crise de la dette souveraine, pandémie de COVID-19, retour brutal de l'inflation, guerre en Ukraine) qui ont modifié la manière dont les banques centrales interviennent.

La Fed et la BCE n'agissent pas dans le même cadre institutionnel. La Fed fonctionne avec un double mandat : stabilité des prix et plein emploi. Lorsqu'elle relève ou abaisse ses taux directeurs, les investisseurs y voient un signe sur la santé de l'économie américaine. Le mandat de la BCE est centré sur la stabilité des prix. Cette différence peut conduire les investisseurs à interpréter les décisions des banques pour ce qu'elles suggèrent sur l'évolution future de l'économie.

L'échantillon couvre les années 2000 à 2024 ce qui permet d'intégrer trois régimes monétaires distincts : la période pré-QE, l'ère des politiques non conventionnelles et la période post-COVID. La politique monétaire n'agit pas directement sur les marchés actions : la littérature identifie quatre grands canaux, le taux d'actualisation, les anticipations, le risque et la liquidité, qui fonctionnent simultanément.

**Question.** Dans quelle mesure les décisions de la BCE et de la Fed influencent-elles les rendements des marchés actions français et américains ?

## II. Données utilisées

Séries publiques mensuelles (FRED, Yahoo Finance, ECB), 297 observations d'avril 2000 à décembre 2024.

- **Variables expliquées** : rendements du CAC 40, du S&P 500 et du Nasdaq 100, de neuf secteurs du CAC 40 (luxe, finance, industrie, technologie, santé, énergie, consommation, automobile, télécoms) et de quatre actions (LVMH, BNP Paribas, Microsoft, Apple). Rendements logarithmiques. Ceux des actions et des secteurs sont calculés à partir des prix ajustés (adjusted close), ce qui intègre les dividendes ; les trois indices sont des indices de prix. Chaque secteur est la moyenne des rendements de ses entreprises.
- **Variables d'intérêt** : taux directeurs de la BCE (taux des opérations principales de refinancement) et de la Fed (cible des fonds fédéraux), taux souverains à 10 ans, variables muettes de régime (pré-2008, post-2008, post-COVID) et de QE.
- **Contrôles** : inflation, chômage, taux de change usd/eur (dollars pour un euro), VIX, prix du pétrole.
- **Extension** : chocs à haute fréquence de Jarociński et Karadi (2020), qui séparent un choc de politique monétaire (taux en hausse, actions en baisse) d'un choc d'information (taux en hausse, actions en hausse).

## III. Méthodologie

Les tests de stationnarité (ADF, KPSS) et l'examen des fonctions d'autocorrélation précèdent l'estimation. Les modèles sont estimés par les moindres carrés ordinaires (MCO) avec des erreurs standards robustes à l'hétéroscédasticité (HC3). Cinq modèles emboîtés sont estimés pour chaque indice : (1) le taux directeur seul, (2) avec taux long, volatilité et régimes, (3) avec les variables macroéconomiques, (4) avec les variables étrangères, (5) comme le (4) sur le rendement en excès du taux sans risque.

r(i,t) = α + β₁ · taux directeur(t-k) + β₂ · Δ taux long(t) + β₃ · régimes(t) + β₄ · QE(t) + γ' X(t) + ε(t), avec k = 2 pour la BCE.

Plusieurs tests, menés au seuil de 10 %, évaluent la validité des modèles : normalité des résidus (Jarque-Bera), spécification (RESET), multicolinéarité (VIF), hétéroscédasticité (Breusch-Pagan), autocorrélation (Breusch-Godfrey), effets ARCH et observations influentes (distance de Cook). Leurs résultats font partie des résultats (section V). La dynamique des effets est étudiée par des fenêtres roulantes de 84 mois, puis la politique monétaire est mesurée par les surprises d'annonce.

## IV. Résultats

### 1. Taux directeurs, taux longs, volatilité et dollar

Modèle le plus complet (rendement en excès, 297 mois, erreurs HC3 ; * 10 %, ** 5 %, *** 1 %) :

| | Taux BCE (retard 2) | Taux Fed | OAT 10 ans | Treasury 10 ans | VIX | USD/EUR | R² |
|---|---|---|---|---|---|---|---|
| **CAC 40** | −0,014 *** | +0,007 *** | −0,076 *** | +0,088 *** | −0,14 *** | +0,23 ** | 0,50 |
| **S&P 500** | −0,008 *** | +0,006 *** | −0,048 ** | +0,049 *** | −0,14 *** | +0,40 *** | 0,61 |
| **Nasdaq 100** | −0,009 * | non significatif | −0,083 *** | +0,080 *** | −0,17 *** | +0,56 *** | 0,43 |

- Le taux directeur de la BCE pèse sur les trois indices. Une hausse de 1 point observée deux mois plus tôt entraîne une baisse du rendement mensuel de 1,4 point de pourcentage pour le CAC 40, 0,8 pour le S&P 500 et 0,9 pour le Nasdaq 100 (significatif à 10 % pour ce dernier), ce qui correspond au canal du taux d'actualisation.
- Le taux de la Fed, pris en niveau, n'a pas d'effet négatif : positif pour le CAC 40 et le S&P 500, non significatif pour le Nasdaq 100. Les deux taux directeurs sont corrélés à 0,72 et se lisent ensemble.
- Les taux longs agissent en sens opposé. À contrôles égaux, une hausse des taux français fait baisser le CAC 40 (−1,3 point par écart-type de la variation mensuelle de l'OAT) et une hausse des taux américains le fait monter (+1,9 point). Les coefficients sont presque symétriques et les deux taux corrélés à 0,74 : les rendements réagissent à l'écart entre les deux mouvements. Résultats du même ordre pour les indices américains.
- Le VIX est le déterminant le plus stable : dans les modèles complets, une hausse de 1 % réduit le rendement mensuel de 0,14 à 0,17 point, et son coefficient est négatif et significatif à 1 % dans toutes les spécifications où il figure.
- Dollar : une hausse de 1 % du taux usd/eur (plus de dollars pour un euro, soit une dépréciation du dollar) est associée à une hausse du rendement mensuel de 0,23 point (CAC 40), 0,40 (S&P 500) et 0,56 (Nasdaq 100).

### 2. L'effet estimé des taux directeurs varie dans le temps

![Fenêtres roulantes](figures/rolling_windows.png)

Fenêtres roulantes de 84 mois, erreurs HAC (part des fenêtres où le coefficient est significatif à 10 %) :

| Coefficient | Fenêtres finissant 2007-2015 | 2016-2021 | 2022-2024 |
|---|---|---|---|
| Taux BCE, CAC 40 | 95 % (médiane −1,2) | 11 % (−0,7) | 33 % (−0,05) |
| Taux Fed, S&P 500 | 28 % | 6 % | 0 % |
| Taux Fed, Nasdaq 100 | 19 % | 10 % | 3 % |
| OAT 10 ans, CAC 40 | 28 % (médiane +3,1) | 0 % (−1,6) | 33 % (−2,9) |
| Treasury 10 ans, S&P 500 | 74 % | 99 % | 6 % |
| Treasury 10 ans, Nasdaq 100 | 97 % | 68 % | 31 % |
| VIX (les trois marchés) | 100 % | 100 % | 100 % |

Pour le taux BCE, les fenêtres significatives de 2022-2024 comptent 22 % de coefficients négatifs (fenêtres se terminant de janvier à août 2022) et 11 % de coefficients positifs (novembre 2022 à mars 2023). Le détail par fenêtre est dans `sorties/rolling_coefficients.csv`.

- Taux BCE et CAC 40. Le coefficient est négatif et significatif dans 95 % des fenêtres se terminant entre 2007 et 2015 (médiane de −1,2 point). Il ne l'est plus que dans 5 % des fenêtres se terminant entre janvier 2016 et novembre 2020. Dans celles qui se terminent entre décembre 2020 et août 2022, il devient très négatif (de −3 à −46) et significatif dans 62 % des cas, mais le taux n'y prend que deux à quatre valeurs, entre 0 et 0,25 % : ces estimations reposent sur des variations de quelques points de base et ne mesurent pas une sensibilité plus forte. Quand le taux recommence à varier, le coefficient est positif jusqu'en mars 2023 (de +1 à +3, significatif dans quatre fenêtres), puis proche de zéro.
- Taux de la Fed. Il est significatif dans 28 % (S&P 500) et 19 % (Nasdaq 100) des fenêtres jusqu'en 2015, et dans 10 % au plus ensuite. Les valeurs très négatives des fenêtres se terminant entre fin 2014 et début 2017 ont la même origine : le taux est resté à 0,125 % de 2009 à 2015.
- Taux longs et VIX. Le coefficient de l'OAT est instable (positif jusqu'en 2015, négatif ensuite). Celui du Treasury à 10 ans est positif et significatif dans la plupart des fenêtres jusqu'en 2021, puis change de signe. Seul le VIX est significatif et négatif dans toutes les fenêtres.

### 3. Secteurs et actions

Même spécification, rendement de chaque série (erreurs HC3) :

| | Taux BCE | Taux Fed | OAT 10 ans | Treasury 10 ans | VIX | USD/EUR |
|---|---|---|---|---|---|---|
| Luxe | −0,015 *** | +0,008 ** | −0,042 | +0,034 | −0,14 *** | +0,02 |
| Finance | −0,014 ** | +0,008 * | −0,126 ** | +0,169 *** | −0,19 *** | +0,59 *** |
| Industrie | −0,010 ** | +0,004 | −0,094 ** | +0,099 *** | −0,14 *** | +0,05 |
| Technologie | −0,016 ** | +0,007 | −0,112 *** | +0,159 *** | −0,18 *** | +0,27 |
| Santé | −0,007 * | +0,006 ** | −0,044 * | +0,019 | −0,08 *** | −0,29 ** |
| Énergie | −0,014 *** | +0,009 ** | −0,084 ** | +0,090 *** | −0,11 *** | +0,24 |
| Consommation | −0,011 *** | +0,007 *** | −0,035 | +0,034 * | −0,09 *** | −0,04 |
| Automobile | −0,022 *** | +0,015 *** | −0,108 ** | +0,124 *** | −0,14 *** | +0,60 ** |
| Télécoms | −0,005 | +0,000 | −0,112 *** | +0,162 *** | −0,11 *** | +0,47 * |
| BNP Paribas | −0,013 * | +0,009 * | −0,147 *** | +0,172 *** | −0,16 *** | +0,42 * |
| LVMH | −0,017 *** | +0,005 | −0,039 | +0,054 * | −0,16 *** | +0,08 |
| Microsoft | −0,004 | +0,003 | −0,080 ** | +0,084 *** | −0,14 *** | +0,59 ** |
| Apple | −0,017 ** | −0,001 | −0,154 *** | +0,083 * | −0,17 *** | +0,58 |

Le taux de la BCE est significativement négatif pour 11 des 13 séries (hors télécoms et Microsoft), avec l'effet le plus fort pour l'automobile. La finance, l'automobile, la technologie, les télécoms et BNP Paribas ont les coefficients les plus élevés sur les deux taux longs (Apple a le plus élevé sur l'OAT seule). Le taux directeur de la Fed n'a un coefficient négatif pour aucune série française : l'influence américaine sur les actions françaises passe par les taux longs américains, le dollar et le VIX.

### 4. Prolongement : les surprises d'annonce

Un niveau de taux étant connu à l'avance par les marchés, la politique monétaire est aussi mesurée par les chocs de Jarociński et Karadi. Rendement mensuel (%) pour un choc d'un écart-type, 297 mois, erreurs HAC :

| Indice | Politique BCE | Information BCE | Politique Fed | Information Fed | R² |
|---|---|---|---|---|---|
| **CAC 40** | −0,69 ** | +1,09 *** | −0,53 ** | +0,60 ** | 0,10 |
| **S&P 500** | −0,40 | +0,61 ** | −0,83 *** | +0,50 * | 0,09 |
| **Nasdaq 100** | −0,22 | +0,76 ** | −1,53 *** | +0,55 | 0,08 |

- Chaque marché réagit d'abord à sa propre banque centrale.
- La transmission est asymétrique : le choc de la Fed pèse aussi sur le CAC 40 (−0,5 %), pas celui de la BCE sur les indices américains.
- Le Nasdaq 100, dominé par les valeurs de croissance, réagit près de deux fois plus que le S&P 500.
- Les chocs d'information font monter les actions (+0,5 à +1,1 %) et brouillent l'effet du niveau des taux.
- Par secteur, un choc de la BCE fait baisser l'énergie (−1,2 %), l'automobile (−1,3 %), l'industrie (−0,9 %), la santé (−0,7 %), la consommation (−0,5 %) et BNP Paribas (−1,6 %) ; un choc de la Fed fait baisser la technologie (−1,7 %), le luxe (−1,0 %), LVMH (−1,2 %), les télécoms (−0,8 %), Microsoft (−1,4 %) et Apple (−1,6 %).

![Secteurs et actions](figures/sectors_and_stocks.png)

## V. Limites des MCO

Ces limites font partie des résultats. Tests de validité du modèle complet, au seuil de 10 % (p-value entre parenthèses) :

| Indice | Normalité (Jarque-Bera) | Forme fonctionnelle (RESET) | Homoscédasticité (Breusch-Pagan) | Absence d'autocorrélation (Breusch-Godfrey) | Absence d'effets ARCH |
|---|---|---|---|---|---|
| CAC 40 | rejetée (0,09) | non rejetée (0,18) | rejetée (0,055) | rejetée (0,02) | rejetée (0,001) |
| S&P 500 | rejetée (< 0,001) | non rejetée (0,62) | rejetée (< 0,001) | non rejetée (0,11) | rejetée (< 0,001) |
| Nasdaq 100 | rejetée (< 0,001) | rejetée (0,036) | rejetée (< 0,001) | non rejetée (0,32) | rejetée (< 0,001) |

Chaque colonne donne l'hypothèse testée : « rejetée » signale un problème.

- **Hypothèses non vérifiées.** Les résidus ne sont normaux pour aucun indice (queues épaisses), leur variance n'est pas constante, des effets ARCH (regroupement de la volatilité) apparaissent partout, une autocorrélation subsiste pour le CAC 40 et la forme fonctionnelle est rejetée pour le Nasdaq 100. La multicolinéarité est faible (VIF maximal de 4,2) et aucune observation n'est influente (distance de Cook maximale de 0,15).
- **Séries persistantes.** Trois variables explicatives sont en niveau : les deux taux directeurs et le taux de chômage américain. Pour chacune, le test ADF rejette la racine unitaire (p = 0,05 pour le taux BCE, 0,01 pour le taux Fed, 0,04 pour le chômage) et le test KPSS rejette la stationnarité (p ≤ 0,01, 0,07 et 0,06) : leur forte persistance fragilise l'inférence.
- **Conséquences pour l'inférence.** L'hétéroscédasticité, l'autocorrélation et la non-normalité laissent les estimateurs MCO convergents, mais les écarts-types classiques ne sont plus fiables : les résultats sont présentés avec des erreurs robustes (HC3). Pour le Nasdaq 100, le rejet du test RESET indique que la forme linéaire est une approximation. Le résultat principal résiste : le taux de la BCE reste significatif à 5 % pour le CAC 40 (p < 0,001 en HC3 comme en HAC) et le S&P 500 (p = 0,005 en HC3, 0,008 en HAC), et autour de 5 % pour le Nasdaq 100 (p = 0,05 en HC3, 0,03 en HAC).
- **Conséquences pour l'interprétation.** Les MCO mesurent des corrélations conditionnelles, pas des effets causaux : le taux directeur réagit lui-même à la conjoncture et aux marchés, et le VIX ou la variation des taux longs, observés le même mois, réagissent aux mêmes nouvelles. Les coefficients moyens sur 25 ans masquent des régimes différents, et les données mensuelles diluent la réaction à l'annonce.

## VI. Conclusion

La politique monétaire influence les marchés actions français et américains, mais selon des mécanismes et une intensité qui varient selon les instruments, les périodes et les secteurs. Le taux directeur de la BCE pèse sur les trois indices et sur la plupart des secteurs, les taux longs agissent en sens opposé en Europe et aux États-Unis, et le VIX est le déterminant le plus stable. L'effet des taux directeurs n'est pas constant dans le temps. Les surprises d'annonce montrent que chaque marché réagit d'abord à sa propre banque centrale et que la Fed se transmet à l'Europe mais pas l'inverse. Ces résultats reposent sur des MCO dont plusieurs hypothèses sont rejetées et qui ne permettent pas d'établir de causalité : ils se lisent comme des corrélations conditionnelles robustes.

## VII. Bibliographie

Altavilla C., Brugnolini L., Gürkaynak R., Motto R. et Ragusa G. (2019). Measuring euro area monetary policy. *Journal of Monetary Economics*, 108, 162-179.
Bernanke B. S. et Kuttner K. N. (2005). What Explains the Stock Market's Reaction to Federal Reserve Policy ? *The Journal of Finance*, 60(3), 1221-1257.
Jarociński M. et Karadi P. (2020). Deconstructing monetary policy surprises: the role of information shocks. *American Economic Journal: Macroeconomics*, 12(2), 1-43.
