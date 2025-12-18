import time
import random
from PIL import Image, ImageDraw

# --- IMPORT DU MODULE NLP DE TON GROUPE ---
try:
    import nlp_emo
except ImportError:
    print("ERREUR CRITIQUE : Le fichier nlp_emo.py est introuvable.")
    nlp_emo = None

class AIController:
    def __init__(self):
        print("Initialisation du contrôleur IA...")
        
        # 1. Chargement du Lexique (Une seule fois au démarrage)
        if nlp_emo:
            self.lexicon = nlp_emo.load_lexicon()
            print(f"Lexique chargé : {len(self.lexicon)} catégories.")
        else:
            self.lexicon = {}

    def process_pipeline(self, user_text):
        """
        Pipeline hybride :
        - NLP : Vrai code (nlp_emo.py)
        - Image/Audio : Simulation (en attendant les modules)
        """
        print(f"--- Traitement : {user_text} ---")

        # ====================================================
        # ETAPE 1 : ANALYSE EMOTIONNELLE (VRAI CODE)
        # ====================================================
        if nlp_emo:
            # Analyse complète (Valence, Arousal, Probas...)
            emo_data = nlp_emo.analyze_text_emotion(user_text, self.lexicon)
            
            # Génération du prompt technique via ta fonction
            base_prompt = nlp_emo.emotion_to_prompt(user_text, emo_data)
            
            # On ajoute le tag visuel [emotion:x] pour faire comme ton exemple
            primary_emotion = emo_data.labels[0] if emo_data.labels else "neutre"
            final_prompt = f"[emotion:{primary_emotion}] {base_prompt}"
            
            # Récupération des valeurs pour la simulation visuelle
            valence, arousal = emo_data.va
        else:
            # Fallback si nlp_emo plante
            final_prompt = f"Erreur NLP - {user_text}"
            valence, arousal = 0.5, 0.5


        # ====================================================
        # ETAPE 2 : GENERATION SPECTROGRAMME (SIMULATION)
        # ====================================================
        # Ici brancher plus tard : image = stable_diffusion.generate(final_prompt)
        time.sleep(2) 
        
        # Création d'un faux spectrogramme qui change de couleur selon l'émotion détectée
        # Valence (X) -> Rouge vers Vert
        # Arousal (Y) -> Sombre vers Lumineux
        r = int((1 - valence) * 255)
        g = int(valence * 255)
        b = int(arousal * 255)
        
        img = Image.new('RGB', (512, 256), color=(r, g, b))
        
        # On dessine quelques traits pour faire "spectrogramme"
        draw = ImageDraw.Draw(img)
        for i in range(0, 512, 10):
            height = random.randint(10, 200) * arousal
            draw.line([(i, 256), (i, 256 - height)], fill=(255, 255, 255), width=2)


        # ====================================================
        # ETAPE 3 : RECONSTRUCTION AUDIO (SIMULATION)
        # ====================================================
        # Ici brancher : audio_path = vocoder.reconstruct(img)
        audio_path = "output_generated.wav"

        return final_prompt, img, audio_path