# -*- coding: utf-8 -*-
"""
Module NLP & Analyse Émotionnelle
Ce module transforme une phrase utilisateur en :
1. Une distribution de probabilités d'émotions.
2. Un couple Valence/Arousal (Psychologie).
3. Un prompt technique pour Stable Diffusion Audio.
"""

from collections import defaultdict
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 1. CONFIGURATION & LEXIQUE ÉTENDU
# =============================================================================

# Mapping Emotion -> (Valence, Arousal)
# Valence : 0.0 (Négatif/Désagréable) à 1.0 (Positif/Agréable)
# Arousal : 0.0 (Calme/Passif) à 1.0 (Excitant/Actif)
EMO_TO_VA = {
    "joie":      (0.90, 0.70), # Très positif, assez actif
    "tristesse": (0.15, 0.20), # Très négatif, très passif
    "colere":    (0.10, 0.90), # Très négatif, très actif
    "calme":     (0.60, 0.10), # Légèrement positif, très passif
    "mystere":   (0.45, 0.50), # Neutre, activité moyenne
    "energie":   (0.75, 0.95), # Positif, très actif
    "peur":      (0.20, 0.80)  # Négatif, très actif
}

# Lexique enrichi (~300 mots)
# Poids 1 = mot standard, Poids 2 = mot fort, Poids 3 = mot déterminant
DEFAULT_LEXICON = {
    "joie": {
        "heureux": 2, "heureuse": 2, "content": 2, "contente": 2, "joie": 3,
        "bonheur": 3, "sourire": 2, "rire": 2, "fete": 2, "amusement": 2,
        "super": 1, "genial": 2, "top": 1, "excellent": 2, "adore": 2,
        "plaisir": 2, "ravi": 2, "enjoue": 2, "magnifique": 2, "succes": 2,
        "gagner": 2, "victoire": 2, "paradisiaque": 2, "soleil": 1, "beau": 1,
        "belle": 1, "merveilleux": 2, "cool": 1, "fun": 1, "positif": 1,
        "amour": 2, "passion": 2, "aime": 1, "celebrer": 2, "enthousiasme": 3
    },
    "tristesse": {
        "triste": 3, "tristesse": 3, "pleurer": 3, "larmes": 2, "chagrin": 3,
        "malheureux": 2, "seul": 2, "solitude": 2, "abandon": 2, "perdu": 2,
        "deception": 2, "decu": 2, "gris": 1, "sombre": 1, "pluie": 1,
        "nostalgie": 2, "melancolie": 3, "deprime": 3, "desespoir": 3,
        "douleur": 2, "souffrance": 2, "regret": 2, "echec": 2, "fatigue": 1,
        "las": 1, "vide": 2, "coeur brise": 3, "adieu": 2, "mort": 2, "deuil": 3
    },
    "colere": {
        "colere": 3, "rage": 3, "furieux": 3, "enerve": 2, "fache": 2,
        "haine": 3, "deteste": 2, "insupportable": 2, "cri": 2, "hurler": 2,
        "agressif": 2, "violence": 3, "bagarre": 2, "guerre": 2, "conflit": 2,
        "tuer": 3, "frapper": 2, "idiot": 1, "stupide": 1, "merde": 2,
        "putain": 2, "vengeance": 3, "jalousie": 2, "frustration": 2,
        "tension": 1, "ennemi": 2, "revolte": 2, "brutal": 2
    },
    "calme": {
        "calme": 3, "paisible": 3, "zen": 3, "tranquille": 2, "repos": 2,
        "silence": 2, "doux": 2, "douceur": 2, "lent": 1, "lentement": 1,
        "dormir": 2, "reve": 2, "nuit": 1, "serein": 3, "apaisant": 3,
        "relax": 2, "detente": 2, "meditation": 3, "nature": 1, "foret": 1,
        "riviere": 1, "brise": 1, "harmonie": 2, "confort": 1, "placide": 2
    },
    "mystere": {
        "mystere": 3, "etrange": 2, "bizarre": 1, "inconnu": 2, "secret": 2,
        "enigme": 2, "suspect": 1, "doute": 1, "ombre": 2, "cache": 1,
        "brume": 2, "brouillard": 2, "fantome": 2, "esprit": 1, "magie": 1,
        "sorcier": 1, "cosmos": 1, "univers": 1, "profond": 1, "lointain": 1,
        "flou": 1, "invisible": 2, "suspense": 2, "tension": 1, "curieux": 1
    },
    "energie": {
        "energie": 3, "force": 2, "puissant": 2, "fort": 1, "rapide": 2,
        "vite": 2, "vitesse": 2, "courir": 2, "sauter": 1, "danser": 2,
        "bouger": 1, "sport": 2, "action": 2, "explosif": 3, "boom": 2,
        "rythme": 2, "intense": 2, "electrique": 2, "fou": 1, "dingue": 1,
        "frenesie": 3, "adrenaline": 3, "motivé": 2, "determination": 2
    },
    "peur": {
        "peur": 3, "crainte": 2, "effraye": 3, "terreur": 3, "horreur": 3,
        "panique": 3, "danger": 2, "mortel": 2, "monstre": 2, "cauchemar": 2,
        "anxiete": 2, "stress": 2, "nervous": 1, "fuir": 2, "cri": 1,
        "glace": 1, "sombre": 1, "effroi": 3, "phobie": 2
    }
}

# =============================================================================
# 2. FONCTIONS UTILITAIRES
# =============================================================================

@dataclass
class EmotionOutput:
    labels: list[str]          
    probs: dict[str, float]    # scores normalisés (somme=1)
    va: tuple[float, float]    # (valence, arousal) calculé
    raw_scores: dict[str, int] # scores bruts pour debug

def normalize(text: str) -> str:
    """Nettoyage strict du texte pour maximiser le matching."""
    text = text.lower()
    # Suppression accents
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    # On remplace tout ce qui n'est pas lettre/chiffre par un espace (gère la ponctuation)
    text = re.sub(r"[^a-z0-9]", " ", text)
    # On compacte les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_lexicon(path: str | Path | None = None) -> dict:
    """Charge un lexique externe JSON si fourni, sinon utilise celui par défaut."""
    if path and Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Echec chargement lexique {path}: {e}")
    return DEFAULT_LEXICON

def softmax_dict(d: dict[str, float]) -> dict[str, float]:
    """Transformation Softmax pour obtenir des probabilités."""
    import math
    if not d: return {}
    # On soustrait le max pour stabilité numérique
    m = max(d.values())
    try:
        exps = {k: math.exp(v - m) for k, v in d.items()}
        s = sum(exps.values())
        return {k: (val/s) for k, val in exps.items()}
    except OverflowError:
        return {k: 1.0/len(d) for k in d} 

def aggregate_va(probs: dict[str, float]) -> tuple[float,float]:
    """Calcule la Valence/Arousal moyenne pondérée par les probabilités."""
    if not probs: return (0.5, 0.5) 
    
    val_sum = 0.0
    aro_sum = 0.0
    
    for emo, prob in probs.items():
        v, a = EMO_TO_VA.get(emo, (0.5, 0.5))
        val_sum += v * prob
        aro_sum += a * prob
        
    return (val_sum, aro_sum)

# =============================================================================
# 3. FONCTIONS PRINCIPALES (API)
# =============================================================================

def analyze_text_emotion(text: str, lexicon: dict = None) -> EmotionOutput:
    """
    Analyse le texte et retourne l'objet EmotionOutput complet.
    C'est la fonction principale appelée par le contrôleur.
    """
    if lexicon is None: lexicon = DEFAULT_LEXICON
        
    t = normalize(text)
    scores = defaultdict(int)
    
    # Matching par mots entiers
    for emo, words in lexicon.items():
        for w, weight in words.items():
            if f" {w} " in f" {t} ": 
                scores[emo] += weight

    if not scores:
        scores = {"calme": 1}

    probs = softmax_dict(scores)
    va = aggregate_va(probs)
    labels = [k for k,_ in sorted(probs.items(), key=lambda kv: -kv[1])[:2]]
    
    return EmotionOutput(labels=labels, probs=probs, va=va, raw_scores=dict(scores))

def emotion_to_prompt(user_text: str, emo: EmotionOutput) -> str:
    """
    Génère le prompt technique pour Stable Diffusion.
    Format : [Description] + [Paramètres techniques] + [Valence/Arousal]
    """
    v, a = emo.va
    mood_str = ", ".join(emo.labels)
    
    # Nettoyage de la phrase utilisateur pour éviter injection de prompt
    safe_text = user_text.replace("\n", " ").strip()
    
    return (
        f"{safe_text}. "
        f"spectrogram of {mood_str} ambient sound, sustained tones, minimal rhythm. "
        f"valence:{v:.2f}, arousal:{a:.2f}, 24kHz, 10s, clean texture"
    )