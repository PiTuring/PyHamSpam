import numpy as np
import pickle
import os
import re

# Définition des constantes 
TAILLE_MIN_MOT = 3
EPSILON = 1


def lireMail(fichierMail: str, dictionnaire: list) -> np.ndarray :
	"""
	Fonction qui lit un fichier mail `fichierMail` et retourne un vecteur de booléens en fonction 
	du dictionnaire.

	Ex. 
		Si le mot de coordonné i dans le dictionnaire `dictionnaire` est présent dans `fichierMail`
		alors x[i] = True, sinon x[i] = False

	Args:
		fichierMail 	(str)	: Nom du fichier mail à lire.
		dictionnaire 	(list)	: Dictionnaire utilisé pour création du vecteur x.

	Returns:
		np.ndarray				: Le vecteur booléen d'appartenance
	"""
	f = open(fichierMail, "r",encoding="ascii", errors="surrogateescape")

	# On extrait les mots du mail :
	texte = f.read().upper() # Mise en MAJ pour insensiblité à la casse
	f.close()
	mots = re.findall(r'\b[A-Z]+\b', texte) # cela ignore la ponctuation

	# Construction du vecteur :
	x = np.isin(dictionnaire, mots) 
	
	return x

def charge_dico(fichierDico: str) -> list:
	"""
	Fonction qui insert dans une liste `mots` tous les mots d'un fichier
	`fichierDico` dictionnaire (.txt) donné en paramètre.
	Les mots de moins de 3 lettres sont ignorés.

	Args:
		fichierDico (str)	: Le chemin vers le fichier dictionnaire.

	Returns:
		list				: La liste de tous les mots du dictionnaire.
	"""
	f = open(fichierDico, "r")
	texte = f.read().upper()
	f.close()

	# On extrait les mots de 3 lettres au moins :
	mots = [mot for mot in re.findall(r'\b[A-Z]+\b', texte) if len(mot) >= TAILLE_MIN_MOT] # Mise en MAJ pour insensibilité à la casse

	print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
	
	return mots[:-1]

def apprendBinomial(dossier: str, fichiers: list, dictionnaire: list) -> np.ndarray:
	"""
	Fonction d'apprentissage d'une loi binomiale à partir des fichiers d'un dossier.
	Retourne un vecteur b de paramètres qui, pour chaque j, calcul la fréquence d'apparition.

	Args:
		dossier 		(str)	: Nom du dossier dans lequel piocher les fichiers de la base d'apprentissage.
		fichiers 		(list)	: Liste des fichiers d'apprentissage.
		dictionnaire 	(list)	: Dictionnaire utilisé pour l'apprentissage.

	Returns:
		np.ndarray				: Vecteur de paramètres
	"""
	N = len(fichiers) # On récupère le nombre total de fichier dans la base

	# Initialisation du vecteur b de même taille que le dictionnaire :
	b = np.zeros(len(dictionnaire))

	# b[j] = nombre de fichier contenant le mot dictionnaire[j] / nombre de fichier

	# Parcours des fichiers un à un comme le sujet le précise :
	for fichier in fichiers:
		# On accumule dans b les vecteurs (comme True = 1, on compte enft le nombre de fichier où apparait chaque mot du dictionnaire)
		b += lireMail(dossier + '/' + fichier, dictionnaire)

	# on utilise epsilon pour eviter que log soit 0
	return (b + EPSILON) / (N + 2 * EPSILON)


def prediction(x: np.ndarray, Pspam: float, Pham: float, bspam: np.ndarray, bham: np.ndarray) -> tuple[bool, float, float]:
	"""
	Fonction qui prédit si un mail représenté par un vecteur booléen `x` est un spam à partir
	du modèle de paramètres `Psam`, `Pham`, `bspam`et `bham`.

	Args:
		x 			(np.ndarray)	: Vecteur booléen représentant un mail.
		Pspam 		(float)			:
		Pham 		(float)			: 
		bspam 		(np.ndarray)	:
		bham 		(np.ndarray)	: 

	Returns:
		tuple[bool, float, float]: 	True si le mail donné en paramètre est classé comme SPAM par le classifieur, False sinon.
									Renvoie aussi les proba calculées.
	"""

	# Pour éviter les problèmes numériques, on passe aux logs :
	# On utilise aussi x et ~x (opérateur NOT numpy) puisque valeur booléennes
	log_spam = np.log(Pspam) + np.sum(x * np.log(bspam) + ~x * np.log(1 - bspam))
	log_ham  = np.log(Pham)  + np.sum(x * np.log(bham)  + ~x * np.log(1 - bham))

	max_log = max(log_spam, log_ham)

	prob_spam = np.exp(log_spam - max_log) / (np.exp(log_spam - max_log) + np.exp(log_ham - max_log))
	prob_ham = np.exp(log_ham - max_log) / (np.exp(log_spam - max_log) + np.exp(log_ham - max_log))
	
	return log_spam > log_ham, prob_spam, prob_ham
	
def test(dossier: str, isSpam: bool, Pspam: float, Pham: float, bspam: np.ndarray, bham: np.ndarray) -> float:
	"""
	Fonction qui test le classifieur de paramètres Pspam, Pham, bspam, bham 
	sur tous les fichiers d'un dossier étiquetés comme SPAM si isSpam et HAM sinon
		
	Args:
		dossier (str)		: Le chemin du dossier sur lequel effectué le test.
		isSpam 	(bool)		: Indique si l'on traite des SPAMs ou des HAMs
		Pspam 	(float)		:
		Pham 	(float)		:
		bspam 	(np.ndarray):
		bham 	(np.ndarray):

	Returns:
		float				: L'erreur de test.
	"""
	fichiers = os.listdir(dossier) # Initialisation de la liste des fichiers à parcourir
	erreurs = 0 # Initialisation du compteur d'erreur
	en_tete = "SPAM" if isSpam else "HAM" # En-tête de message 

	# Parcours des fichiers à tester
	for i, fichier in enumerate(fichiers):
		# Ajout du nom de fichier dans message
		nom =  f" {dossier}/{fichier} identifié comme un "

		# Calcul de la prédiction :
		x = lireMail(dossier + "/" + fichier, dictionnaire)
		pred, prob_spam, prob_ham = prediction(x, Pspam, Pham, bspam, bham)
		
		# Ajout de la catégorie prédite dans le message :
		categorie = "SPAM" if pred else "HAM"

		# Erreur de prédiction (XOR sur les deux puisque si pred = 1 et isSpam = 0, on a classé un HAM dans SPAM et si pred = 0 et isSpam = 1, on a classé un SPAm dans HAM):
		erreur = ""
		if pred ^ isSpam:
			erreur = " *** erreur ***" # Ajout au message
			erreurs += 1 # On ajoute l'erreur

		#print(en_tete + nom + categorie + erreur)
		print(f"{en_tete} numéro {i} : P(Y=SPAM | X=x) = {prob_spam}, P(Y=HAM | X=x) = {prob_ham}", end = '')
		print(f" ==> identifié comme un {categorie}{erreur}")

	return erreurs / len(fichiers) # On retourne le taux d'erreur

def testClassifieur(dossier: str, isSpam: bool, classifieur: dict) -> float:
	"""
	Même fonction que test() mais avec encapsulation dans classifieur.

	Args:
		dossier 	(str)	: Le chemin du dossier sur lequel effectué le test.
		isSpam 		(bool)	: Indique si l'on traite des SPAMs ou des HAMs
		classifieur (dict)	: Le classifieur à tester.
		
	Returns:
		float: L'erreur de test.
	"""
	return test(dossier, isSpam, classifieur['Pspam'], classifieur['Pham'], classifieur['bspam'], classifieur['bham'])

def exporterClassifieur(nomFichier: str, classifieur: dict):
	"""
	Procédure qui sauvegarde un classifieur dans un fichier via Pickle.

	Args:
		nomFichier 	(str)	: Le nom que l'on souhaite donner au fichier.
		classifieur (dict)	: Le classifieur à sauvegarder.
	"""
	with open(f"{nomFichier}.pkl", "wb") as f:
		pickle.dump(classifieur, f)

def importerClassifieur(cheminFichier: str) -> dict:
	"""
	Fonction qui charge un classifieur d'après un fichier de sauvegarde.

	Args:
		cheminFichier (str)	: Le chemin vers le fichier sauvegardant le classifieur.

	Returns:
		dict				: Le classifieur chargé.
	"""
	with open(cheminFichier, "rb") as f:
		classifieur = pickle.load(f)

	return classifieur

def miseAJour(classifieur: dict, fichierMail: str, isSpam: bool) -> dict:
	"""
	Fonction qui applique l'apprentissage en ligne au classifieur donné
	avec le contenu du fichier `fichierMail`.

	Args:
		classifieur (dict)	: Le classifieur que l'on met à jour.
		fichierMail (str)	: Le fichier du mail à ajouter.
		isSpam 		(bool)	: Indique si le fichier correspond à un SPAM ou un HAM.

	Returns:
		dict				: Le classifieur mis à jour.
	"""
	x = lireMail(fichierMail, classifieur['dico'])

	if isSpam: # On met a jour les paramètres spam
		m = classifieur['mSpam']
		n = classifieur['bspam'] * (m + 2 * EPSILON) - EPSILON
		classifieur['bspam'] = (n + x + EPSILON) / (m + 1 + 2 * EPSILON)
		classifieur['mSpam'] += 1
	else: # On met à jour les paramètres ham
		m = classifieur['mHam']
		n = classifieur['bham'] * (m + 2 * EPSILON) - EPSILON
		classifieur['bham'] = (n + x + EPSILON) / (m + 1 + 2 * EPSILON)
		classifieur['mHam'] += 1

	# On met à jour les à priori
	total = classifieur['mSpam'] + classifieur['mHam']
	classifieur['Pspam'] = classifieur['mSpam'] / total
	classifieur['Pham'] = classifieur['mHam'] / total

	return classifieur

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

	# Encapsulation (on met dans un dictionnaire tous les params + le dico)
	classifieur = {
		'Pspam': Pspam,
		'Pham':  Pham,
		'bspam': bspam,
		'bham':  bham,
		'mSpam': mSpam,  
		'mHam':  mHam,
		'dico':  dictionnaire
	}


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


# TODO : Faire un exemple avec et sans lissage via EPSILON
# TODO : Tester l'export puis import d'un classifieur
# TODO : Avec le classifieur importé, tester mise a jour avec apprentissage en ligne