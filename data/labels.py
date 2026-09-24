"""Libellés des variables et de leurs modalités (d'après le sujet)."""

TARGET = "Q19A"
WEIGHT = "pm17B"
ID = "A01"

NON_REPONDU = "Non répondu"
NON_CONCERNE = "Non concerné"

PCS = {
    1: "Agriculteur",
    2: "Artisan, commerçant",
    3: "Chef entr. ≥ 10 sal.",
    4: "Cadre, prof. sup.",
    5: "Prof. intermédiaire",
    6: "Employé",
    7: "Ouvrier",
    8: "Sans profession",
    9: "Non concerné",
}
SITUATION_PARENT = {
    1: "Travaille",
    2: "Chômage",
    3: "Au foyer",
    4: "Invalidité",
    5: "Retraite",
    6: "Ne sait pas",
    7: "Non concerné",
}
ALCOOL_PARENT = {
    1: "Jamais",
    2: "Peu / an",
    3: "1-2 / mois",
    4: "≥ 1 / semaine",
    5: "Presque chaque jour",
}
DIFFICULTE = {1: "Non", 2: "Parfois", 3: "Souvent"}

# variable -> (libellé, {code: libellé de la modalité})
VARIABLES = {
    "Q03": ("Sexe", {1: "Homme", 2: "Femme"}),
    "Q04": ("Situation", {1: "Études", 2: "Études arrêtées"}),
    "Q04A": (
        "Situation scolaire",
        {1: "Lycée / collège", 2: "Apprentissage", 3: "Supérieur"},
    ),
    "Q04B": (
        "Situation professionnelle",
        {1: "Sans activité", 2: "Recherche emploi", 3: "Insertion", 4: "Travaille"},
    ),
    "Q05": ("Redoublement", {1: "Non", 2: "Oui"}),
    "Q06A": ("Difficultés pour lire", DIFFICULTE),
    "Q06B": ("Difficultés pour écrire", DIFFICULTE),
    "Q08": (
        "Lieu de vie",
        {
            1: "Chez parent(s)",
            2: "Internat",
            3: "Foyer / fam. accueil",
            4: "Propre logement",
            5: "Autre",
        },
    ),
    "Q08C": (
        "Vie des parents",
        {1: "Ensemble", 2: "Séparés", 3: "Jamais connus", 4: "Décès", 5: "Autre"},
    ),
    "Q09A1": ("Situation du père", SITUATION_PARENT),
    "Q09B1": ("Situation de la mère", SITUATION_PARENT),
    "Q10A1": ("PCS du père", PCS),
    "Q10B1": ("PCS de la mère", PCS),
    "B08A": ("Père boit à la maison", ALCOOL_PARENT),
    "B08B": ("Mère boit à la maison", ALCOOL_PARENT),
}

PREDICTORS = list(VARIABLES)
