# BTP AI — Système RAG pour Documents de Construction

Un système de génération augmentée par récupération (RAG) conçu pour les entreprises BTP. Importez des PDFs et des e-mails, posez des questions en langage naturel, et obtenez des réponses basées uniquement sur vos documents — avec citations des sources.

**Stack :** Flask · Pinecone · Groq (LLaMA 3) · Pinecone Inference Embeddings · Dashboard HTML

🌐 **Démo en ligne :** [https://ai-system-for-construction-data-btp-project-production.up.railway.app](https://ai-system-for-construction-data-btp-project-production.up.railway.app)

---

## Fonctionnalités

- Import de PDFs, DOCX et fichiers TXT
- Ingestion d'e-mails avec métadonnées projet
- Mémoire conversationnelle et contexte persistant
- Multi-query RAG pour améliorer le rappel sémantique
- Endpoint de conformité réglementaire
- Recherche sémantique multilingue (FR / EN / AR)
- Citations automatiques des sources
- Dashboard web interactif


## Structure du projet

```
.
├── app.py              # API Flask — tous les endpoints
├── config.py           # Variables d'environnement et constantes
├── ingest.py           # Orchestration : découpage + embedding + upsert
├── chunker.py          # Découpe le texte en chunks avec chevauchement
├── embeddings.py       # Embeddings Pinecone (par lots de 96)
├── vectorstore.py      # Upsert / requête / suppression dans Pinecone
├── llm.py              # Appel Groq LLM + construction du prompt + filtrage des sources
├── pdf_reader.py       # Extraction du texte page par page depuis les PDFs
├── dashboard.html      # Interface utilisateur
└── .env                # Secrets (ne jamais committer ce fichier)
```

---

## Prérequis

- Python 3.10+
- Un compte [Pinecone](https://www.pinecone.io/) (gratuit)
- Un compte [Groq](https://console.groq.com/) (gratuit)

---

## Installation

**1. Cloner le dépôt**
```bash
git clone https://github.com/Yassir-Essabbahy/AI-System-for-Construction-Data-BTP-Project.git
cd AI-System-for-Construction-Data-BTP-Project
```

**2. Installer les dépendances**
```bash
pip install flask flask-cors pinecone groq pypdf python-dotenv gunicorn
```

**3. Créer le fichier `.env`**

```env
GROQ_API_KEY=votre_clé_groq
PINECONE_API_KEY=votre_clé_pinecone

# Optionnel — valeurs par défaut indiquées
PINECONE_INDEX_NAME=btp-ai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_MODEL=multilingual-e5-large
GROQ_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
MIN_SCORE=0.5
```

**4. Lancer le serveur**
```bash
python app.py
```

Ouvrez ensuite [http://localhost:5000](http://localhost:5000) dans votre navigateur.

---

## Test rapide — Guide pas à pas

### Documents de test

Deux PDFs sont fournis dans le dépôt pour tester le système :

| Fichier | Contenu |
|---|---|
| `Asphalt.pdf` | Composition et propriétés de l'asphalte |
| `Constuction_Specifications.pdf` | Cahier des charges complet : béton, toiture, plomberie, électricité |

---

### Étape 1 — Importer un PDF via le dashboard

1. Ouvrez le dashboard
2. Cliquez sur **Documents** dans la barre latérale gauche
3. Glissez-déposez `Constuction_Specifications.pdf` dans la zone de dépôt, ou cliquez sur **Choisir un fichier**
4. Attendez que le statut passe à **INDEXÉ** ✅

---

### Étape 2 — Poser des questions sur le PDF

Cliquez sur **Chat** et essayez ces questions :

**Béton et matériaux**
> Quelles sont les proportions du béton de classe A ?

> Quelle marque de ciment est spécifiée dans le document ?

> Quelles sont les exigences pour les armatures en acier ?

**Finitions**
> Quels matériaux sont utilisés pour les finitions de sol ?

> Quelle marque de peinture est utilisée pour ce projet ?

**Plomberie et électricité**
> Quelles sont les spécifications de plomberie ?

> Quelles normes électriques doivent être respectées ?

**Question hors sujet — pour tester le mode strict**
> Quel est le prix de l'acier aujourd'hui ?

➡️ Réponse attendue : *"Je n'ai pas suffisamment d'informations dans les documents fournis."*

---

### Étape 3 — Importer des e-mails via l'API

Depuis un terminal Windows (CMD), copiez-collez cette commande en une seule ligne :

```
curl -X POST https://ai-system-for-construction-data-btp-project-production.up.railway.app/ingest-emails -H "Content-Type: application/json" -d "{\"emails\":[{\"subject\":\"Retard chantier\",\"from\":\"client@btp.com\",\"date\":\"2026-05-01\",\"body\":\"Le delai du projet a ete prolonge jusqu en juin en raison de retards de materiaux.\",\"project\":\"Chantier A\",\"lot\":\"Gros Oeuvre\",\"criticite\":\"Haute\"}]}"
```

Réponse attendue :
```json
{ "message": "Emails ingested.", "chunks_written": 1 }
```

Puis posez cette question dans le chat :
> Quel est le statut du Chantier A ?

---

### Ce qu'il faut observer

| Action | Résultat attendu |
|---|---|
| Importez un PDF | Statut **INDEXÉ** dans l'onglet Documents |
| Posez une question sur le PDF | Réponse avec citation `[1]` et étiquette bleue de la source |
| Même page citée deux fois | Une seule étiquette source affichée (déduplication) |
| Question hors sujet | *"Je n'ai pas suffisamment d'informations..."* |
| Question en français | Réponse en français |
| Question en anglais | Réponse en anglais |

---

## Comment ça fonctionne

```
Question de l'utilisateur
          │
          ▼
embed_query()        — convertit la question en vecteur de 1024 dimensions
          │
          ▼
vectorstore.query()  — trouve les 5 chunks les plus similaires dans Pinecone (score ≥ 0.5)
          │
          ▼
llm.answer()         — envoie les chunks comme contexte à LLaMA 3 via Groq
          │
          ▼
filtrage sources     — ne retourne que les sources citées [1][2]… dans la réponse
          │
          ▼
réponse JSON         — réponse + sources dédupliquées
```

Le découpage utilise une fenêtre glissante : chaque chunk fait au maximum 500 caractères, avec 100 caractères de chevauchement pour ne pas perdre le contexte aux frontières entre chunks.

---

### Multi-query RAG

Avant la recherche vectorielle, le système peut générer plusieurs variantes sémantiques de la question utilisateur afin :
- d'améliorer le rappel des documents,
- de réduire les faux négatifs,
- d'augmenter la pertinence des chunks récupérés.

### Mémoire conversationnelle

Le système conserve le contexte des échanges précédents afin de :
- maintenir la continuité conversationnelle,
- relier les décisions et informations projet,
- améliorer la pertinence des réponses.


## Référence de configuration

| Variable | Défaut | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Remplacer par `llama-3.3-70b-versatile` pour plus de qualité |
| `TOP_K` | `5` | Nombre de chunks récupérés par question |
| `MIN_SCORE` | `0.5` | Score de similarité cosinus minimum pour inclure un chunk |
| `CHUNK_SIZE` | `500` | Taille maximale d'un chunk en caractères |
| `CHUNK_OVERLAP` | `100` | Chevauchement entre chunks consécutifs |
| `EMBEDDING_MODEL` | `multilingual-e5-large` | Supporte le français, l'arabe et l'anglais |

---

## Endpoint conformité réglementaire

Le système inclut un endpoint dédié permettant :
- l'analyse de conformité documentaire,
- la vérification réglementaire,
- l'assistance métier BTP,
- l'évaluation de cohérence technique.

Sources compatibles :
- DTU
- Normes NF / EN / ISO
- Documentation technique
- Documents projet internes



## Notes

- Le modèle utilise **uniquement** les documents fournis. Il ne répond pas depuis ses connaissances générales.
- Les statistiques du dashboard sont liées à la session et se réinitialisent au rechargement de la page.
- Le tier gratuit Pinecone supporte jusqu'à 100 000 vecteurs, soit plusieurs centaines de PDFs.