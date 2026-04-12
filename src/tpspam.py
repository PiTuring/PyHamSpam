import numpy as np
import os
import math

TAILLE_MIN_MOT = 3
EPSILON = 1


def lireMail(fichier: str, dictionnaire: list) -> np.ndarray :
	"""
	Fonction qui lit un fichier `fichier` et retourne un vecteur de booléens en fonction 
	du dictionnaire.

	Ex. 
		Si le mot de coordonné i dans le dictionnaire `dictionnaire` est présent dans `fichier`
		alors x[i] = True, sinon x[i] = False

	Returns:
		np.ndarray: Le vecteur booléen d'appartenance
	"""
	f = open(fichier, "r",encoding="ascii", errors="surrogateescape")

	# On extrait les mots du mail :
	mots = [mot.upper() for mot in f.read().split(" ")] # Mise en MAJ pour insensiblité à la casse

	# Construction du vecteur :
	x = np.isin(dictionnaire, mots) 
	
	f.close()
	return x

def charge_dico(fichier: str) -> list:
	"""
	Fonction qui insert dans une liste `mots` tous les mots d'un fichier
	`fichier` dictionnaire (.txt) donné en paramètre.
	Les mots de moins de 3 lettres sont ignorés.

	Args:
		fichier (str): Le chemin vers le fichier dictionnaire.

	Returns:
		list: La liste de tous les mots du dictionnaire.
	"""
	f = open(fichier, "r")

	# On extrait les mots de 3 lettres au moins :
	mots = [mot.upper() for mot in f.read().split("\n") if len(mot) >= TAILLE_MIN_MOT] # Mise en MAJ pour insensibilité à la casse

	print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
	f.close()
	return mots[:-1]

def apprendBinomial(dossier: str, fichiers: list, dictionnaire: list) -> np.ndarray:
	"""
	Fonction d'apprentissage d'une loi binomiale à partid es fichiers d'un dossier.
	Retourne un vecteur b de paramètres qui, pour chaque j, calcul la fréquence d'apparition.

	Returns:
		np.ndarray: Vecteur de paramètres
	"""
	N = len(fichiers) # On récupère le nombre total de fichier dans la base

	# Initialisation du vecteur b de même taille que le dictionnaire :
	b = np.zeros(len(dictionnaire))

	# b[j] = nombre de fichier contenant le mot dictionnaire[j] / nombre de fichier

	# Parcours des fichiers un à un comme le sujet le précise :
	for fichier in fichiers:
		# On accumule dans b les vecteurs (comme True = 1, on compte enft le nombre de fichier où apparait chaque mot du dictionnaire)
		b += lireMail(dossier + '/' + fichier, dictionnaire)

	return b / N



def prediction(x: np.ndarray, Pspam: float, Pham: float, bspam: np.ndarray, bham: np.ndarray) ->bool:
	"""
	Fonction qui prédit si un mail représenté par un vecteur booléen `x` est un spam à partir
	du modèle de paramètres `Psam`, `Pham`, `bspam`et `bham`.


	Returns:
		bool: True si le mail donné en paramètre est classé comme SPAM par le classifieur, False sinon
	"""

	# Pour éviter les problèmes numériques, on passe aux logs :
	# On utilise aussi x et ~x (opérateur NOT numpy) puisque valeur booléennes
	log_spam = np.log(Pspam) + np.sum(x * np.log(bspam) + ~x * np.log(1 - bspam))
	log_ham  = np.log(Pham)  + np.sum(x * np.log(bham)  + ~x * np.log(1 - bham))

	
	return log_spam > log_ham
	
def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	"""
		Test le classifieur de paramètres Pspam, Pham, bspam, bham 
		sur tous les fichiers d'un dossier étiquetés 
		comme SPAM si isSpam et HAM sinon
		
		Retourne le taux d'erreur 
	"""
	fichiers = os.listdir(dossier) # Initialisation de la liste des fichiers à parcourir
	erreurs = 0 # Initialisation du compteur d'erreur
	en_tete = "SPAM" if isSpam else "HAM" # En-tête de message 

	# Parcours des fichiers à tester
	for fichier in fichiers:
		# Ajout du nom de fichier dans message
		nom =  f" {dossier}/{fichier} identifié comme un "

		# Calcul de la prédiction :
		x = lireMail(dossier + "/" + fichier, dictionnaire)
		pred = prediction(x, Pspam, Pham, bspam, bham)
		
		# Ajout de la catégorie prédite dans le message :
		categorie = "SPAM" if pred else "HAM"

		# Erreur de prédiction (XOR sur les deux puisque si pred = 1 et isSpam = 0, on a classé un HAM dans SPAM et si pred = 0 et isSpam = 1, on a classé un SPAm dans HAM):
		erreur = ""
		if pred ^ isSpam:
			erreur = " *** erreur ***" # Ajout au message
			erreurs += 1 # On ajoute l'erreur

		print(en_tete + nom + categorie + erreur)

	return erreurs / len(fichiers) # On retourne le taux d'erreur


############ programme principal ############
if __name__ == "__main__":
	dossier_spams = "utils/bases/baseapp/spam"
	dossier_hams = "utils/bases/baseapp/ham"

	fichiersspams = os.listdir(dossier_spams)
	fichiershams = os.listdir(dossier_hams)

	mSpam = len(fichiersspams)
	mHam = len(fichiershams)

	# Chargement du dictionnaire:
	dictionnaire = charge_dico("utils/dictionnaire1000en.txt")
	print(dictionnaire)

	# Apprentissage des bspam et bham:
	print("apprentissage de bspam...")
	bspam = apprendBinomial(dossier_spams, fichiersspams, dictionnaire)
	print("apprentissage de bham...")
	bham = apprendBinomial(dossier_hams, fichiershams, dictionnaire)

	# Calcul des probabilités a priori Pspam et Pham:
	# Pspam = nb de spam / nb exemple
	Pspam = mSpam / (mSpam + mHam)
	# Pham = 1 - Pspam
	Pham = 1 - Pspam

	print(f"Probabilité à priori de SPAM : {Pspam}\nProbabilité à priori de HAM : {Pham}")


	# Calcul des erreurs avec la fonction test():
	dossier_test_spams = "utils/bases/basetest/spam"
	dossier_test_hams = "utils/bases/basetest/ham"

	fichiers_test_spams = os.listdir(dossier_test_spams)
	fichiers_test_hams = os.listdir(dossier_test_hams)

	mTestSpam = len(fichiers_test_spams)
	mTestHam = len(fichiers_test_hams)

	erreurTestSpam = test(dossier_test_spams, True, Pspam, Pham, bspam, bham)
	erreurTestHam = test(dossier_test_hams, False, Pspam, Pham, bspam, bham)
	print(f"Erreur de test sur {mTestSpam} SPAM 			: {erreurTestSpam * 100:.0f} %")
	print(f"Erreur de test sur {mTestHam} HAM 			: {erreurTestHam * 100:.0f} %")
	print(f"Erreur de test globale sur {mTestSpam + mTestHam} mails		: {(erreurTestSpam * mTestSpam + erreurTestHam * mTestHam) / (mTestSpam + mTestHam) * 100:.0f} %")

# TODO Modifier prediction qui semble bugger avant de passer à affinage