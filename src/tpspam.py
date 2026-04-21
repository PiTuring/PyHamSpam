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
		Pspam 		(float)			: Proba à priori SPAM
		Pham 		(float)			: Proba à priori HAM
		bspam 		(np.ndarray)	: Paramètres appris sur les SPAMs (fréquences)
		bham 		(np.ndarray)	: Paramètres appris sur les HAMs (fréquences)

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
	
def test(dossier: str, isSpam: bool, Pspam: float, Pham: float, bspam: np.ndarray, bham: np.ndarray, trace: bool=False) -> float:
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
		trace 	(bool)		: Pour gérer affichage. Par défaut à False.

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

		if trace:
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


	# Comparaison avec / sans lissage : 
	bspam_sans = np.zeros(len(dictionnaire))
	for fichier in fichiersspams:
		bspam_sans += lireMail(dossier_spams + "/" + fichier, dictionnaire)
	bspam_sans /= mSpam

	bham_sans = np.zeros(len(dictionnaire))
	for fichier in fichiershams:
		bham_sans += lireMail(dossier_hams + "/" + fichier, dictionnaire)
	bham_sans /= mHam

	# On prend très petit nombre proche de 0 pour eviter log(0)
	bspam_sans = np.clip(bspam_sans, 1e-10, 1 - 1e-10)
	bham_sans = np.clip(bham_sans, 1e-10, 1 - 1e-10)

	err_spam_sans = test(dossier_test_spams, True,  Pspam, Pham, bspam_sans, bham_sans, True)
	err_ham_sans  = test(dossier_test_hams,  False, Pspam, Pham, bspam_sans, bham_sans, True)

	print(f"SANS lissage - SPAM : {err_spam_sans * 100:.0f}%  |  HAM : {err_ham_sans * 100:.0f}%")
	print(f"AVEC lissage - SPAM : {erreurTestSpam * 100:.0f}%  |  HAM : {erreurTestHam * 100:.0f}%")

	# Import / export du classifieur

	exporterClassifieur("mon_classifieur", classifieur)
	classifieur_charge = importerClassifieur("mon_classifieur.pkl")

	err_spam_charge = testClassifieur(dossier_test_spams, True, classifieur_charge)
	err_ham_charge = testClassifieur(dossier_test_hams, False, classifieur_charge)

	print(f"Vérification après import - SPAM : {err_spam_charge * 100:.0f}%  |  HAM : {err_ham_charge * 100:.0f}%")

	# Apprentissage en ligne :
	print(f"Avant mise à jour - mSpam : {classifieur_charge['mSpam']}, mHam : {classifieur_charge['mHam']}")
	print(f"Avant mise à jour - Pspam : {classifieur_charge['Pspam']:.4f}, Pham : {classifieur_charge['Pham']:.4f}")

	nouveau_spam = dossier_test_spams + "/" + fichiers_test_spams[0]
	nouveau_ham = dossier_test_hams + "/" + fichiers_test_hams[0]

	classifieur_charge = miseAJour(classifieur_charge, nouveau_spam, True)
	classifieur_charge = miseAJour(classifieur_charge, nouveau_ham, False)

	print(f"Après mise à jour - mSpam : {classifieur_charge['mSpam']}, mHam : {classifieur_charge['mHam']}")
	print(f"Après mise à jour - Pspam : {classifieur_charge['Pspam']:.4f}, Pham : {classifieur_charge['Pham']:.4f}")
 
	err_spam_maj = testClassifieur(dossier_test_spams, True,  classifieur_charge)
	err_ham_maj  = testClassifieur(dossier_test_hams,  False, classifieur_charge)

	print(f"Après mise à jour - SPAM : {err_spam_maj * 100:.0f}%  |  HAM : {err_ham_maj * 100:.0f}%")
	
	# Apprentissage en ligne sur plusieurs mails :

	N_INIT = 300
	N_MAX_HAM = 1200
	PAS = 50

	spams_init = fichiersspams[:N_INIT]
	hams_init = fichiershams[:N_INIT]
	bspam = apprendBinomial(dossier_spams, spams_init, dictionnaire)
	bham = apprendBinomial(dossier_hams, hams_init, dictionnaire)
	Pspam = N_INIT / (N_INIT + N_INIT)
	Pham = 1 - Pspam
	classifieur = {
		'Pspam': Pspam,
		'Pham':  Pham,
		'bspam': bspam,
		'bham':  bham,
		'mSpam': N_INIT,
		'mHam':  N_INIT,
		'dico':  dictionnaire
	}

	def erreur_globale(classifieur, dossier_test_spams, dossier_test_hams):
		fichiers_s = os.listdir(dossier_test_spams)
		fichiers_h = os.listdir(dossier_test_hams)
		erreurs_s = sum(
			1 for f in fichiers_s
			if not prediction(lireMail(dossier_test_spams + '/' + f, classifieur['dico']),
							  classifieur['Pspam'], classifieur['Pham'],
							  classifieur['bspam'], classifieur['bham'])[0]
		)
		erreurs_h = sum(
			1 for f in fichiers_h
			if prediction(lireMail(dossier_test_hams + '/' + f, classifieur['dico']),
						  classifieur['Pspam'], classifieur['Pham'],
						  classifieur['bspam'], classifieur['bham'])[0]
		)
		total = len(fichiers_s) + len(fichiers_h)
		return erreurs_s / len(fichiers_s), erreurs_h / len(fichiers_h), (erreurs_s + erreurs_h) / total
 
	err_s, err_h, err_g = erreur_globale(classifieur, dossier_test_spams, dossier_test_hams)

	# Pour graphe
	x_points = [0] # initial
	errs_spam = [err_s * 100]
	errs_ham = [err_h * 100]
	errs_glob = [err_g * 100]

	spams_restants = fichiersspams[N_INIT:]
	hams_restants = fichiershams[N_INIT:N_MAX_HAM]
	mails_a_ajouter = []
	for s, h in zip(spams_restants, hams_restants):
		mails_a_ajouter.append((dossier_spams + "/" + s, True))
		mails_a_ajouter.append((dossier_hams + "/" + h, False))

	for s in spams_restants[len(hams_restants):]:
		mails_a_ajouter.append((dossier_spams + '/' + s, True))
	for h in hams_restants[len(spams_restants):]:
		mails_a_ajouter.append((dossier_hams  + '/' + h, False))

	

	for i, (chemin_mail, isSpam) in enumerate(mails_a_ajouter, start=1):
		classifieur = miseAJour(classifieur, chemin_mail, isSpam)

		# On met à jours les erreurs pour suivi
		if i % PAS == 0 or i == len(mails_a_ajouter):
			err_s, err_h, err_g = erreur_globale(classifieur, dossier_test_spams, dossier_test_hams)
			x_points.append(i)
			errs_spam.append(err_s * 100)
			errs_ham.append(err_h  * 100)
			errs_glob.append(err_g * 100)

	# Affichage : 
	import matplotlib.pyplot as plt
 
	plt.figure(figsize=(10, 6))
	plt.plot(x_points, errs_spam, marker='o', color='crimson',    label='Erreur SPAM')
	plt.plot(x_points, errs_ham,  marker='s', color='steelblue',  label='Erreur HAM')
	plt.plot(x_points, errs_glob, marker='^', color='darkorange', label='Erreur globale', linestyle='--')
 
	plt.axvline(x=0, color='gray', linestyle=':', linewidth=1, label=f'Départ : {N_INIT} spams + {N_INIT} hams')
 
	plt.title("Évolution du taux d'erreur - apprentissage en ligne", fontsize=14)
	plt.xlabel("Nombre de mails ajoutés en ligne", fontsize=12)
	plt.ylabel("Taux d'erreur (%)", fontsize=12)
	tick_labels = [str(x) if i % 5 == 0 else '' for i, x in enumerate(x_points)]
	plt.xticks(x_points, tick_labels, rotation=45)
	plt.ylim(0, 100)
	plt.legend()
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.show()