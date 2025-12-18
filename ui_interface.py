import customtkinter as ctk
from PIL import Image
import threading
import time
import datetime 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from backend_logic import AIController

# --- CONFIGURATION DU THEME ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class AppInterface(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- SETUP FENETRE PRINCIPALE ---
        self.title("Polytech AI Audio Studio")
        self.geometry("1350x850") 
        self.configure(fg_color="#121212") 

        self.ai = AIController()
        self.is_admin_unlocked = False
        self.is_playing = False
        self.view_mode = "COVER"
        self.session_history = [] # Pour stocker les données
        self.grid_columnconfigure(0, weight=0, minsize=260) 
        self.grid_columnconfigure(1, weight=1) 
        
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Visual
        self.grid_rowconfigure(2, weight=0) # Player
        self.grid_rowconfigure(3, weight=0) # Input
        self.grid_rowconfigure(4, weight=0) # Console

        # =================================================
        # 0. SIDEBAR (HISTORIQUE) - GAUCHE
        # =================================================
        self.sidebar_frame = ctk.CTkFrame(self, fg_color="#181818", corner_radius=0, width=260)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_propagate(False) 

        # Titre Sidebar
        ctk.CTkLabel(self.sidebar_frame, text="HISTORIQUE", font=("Arial", 14, "bold"), text_color="#666666").pack(pady=(30, 20), padx=20, anchor="w")

        # Liste déroulante
        self.scroll_history = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="transparent")
        self.scroll_history.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Message vide par défaut
        self.lbl_empty_hist = ctk.CTkLabel(self.scroll_history, text="Aucune génération...", text_color="#444", font=("Arial", 12, "italic"))
        self.lbl_empty_hist.pack(pady=20)


        # =================================================
        # 1. HEADER (Colonne 1)
        # =================================================
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=1, sticky="ew", padx=30, pady=(20, 0))
        
        lbl_title = ctk.CTkLabel(self.header_frame, text="AI AUDIO STUDIO", 
                                 font=("Arial", 14, "bold"), text_color="#b3b3b3")
        lbl_title.pack(side="left")

        # Boutons droite
        self.btn_admin = ctk.CTkButton(self.header_frame, text="⚙", width=40, height=40,
                                       fg_color="transparent", text_color="#666666",
                                       hover_color="#222222", font=("Arial", 24),
                                       command=self.toggle_admin_panel)
        self.btn_admin.pack(side="right")

        self.btn_view = ctk.CTkButton(self.header_frame, text="📊 Signal", width=100, height=30,
                                      fg_color="#333333", hover_color="#444444",
                                      font=("Arial", 12), command=self.toggle_view_mode)
        self.btn_view.pack(side="right", padx=10)


        # =================================================
        # 2. ZONE VISUELLE (Colonne 1)
        # =================================================
        self.visual_container = ctk.CTkFrame(self, fg_color="transparent")
        self.visual_container.grid(row=1, column=1, sticky="nsew", padx=50, pady=20)
        self.visual_container.grid_columnconfigure(0, weight=1)
        self.visual_container.grid_rowconfigure(0, weight=1)

        # MODE A : COVER
        self.cover_frame = ctk.CTkFrame(self.visual_container, width=400, height=400, 
                                        fg_color="#1e1e1e", corner_radius=15)
        self.cover_frame.grid(row=0, column=0)
        self.cover_frame.grid_propagate(False)

        self.lbl_image = ctk.CTkLabel(self.cover_frame, text="No Audio Generated", 
                                      font=("Arial", 16), text_color="#555555")
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")

        # MODE B : GRAPHES 
        self.graph_frame = ctk.CTkFrame(self.visual_container, fg_color="transparent")
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4), facecolor='#121212')
        self.fig.subplots_adjust(wspace=0.25, left=0.10, right=0.95, top=0.85, bottom=0.25)
        
        for ax in self.axs:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(axis='x', colors='#888888', labelsize=9)
            ax.tick_params(axis='y', colors='#888888', labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#333333')
            ax.grid(True, color='#333333', linestyle='--', alpha=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both", pady=10)


        # =================================================
        # 3. PLAYER & INPUTS (Colonne 1)
        # =================================================
        self.info_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.info_frame.grid(row=2, column=1, pady=(0, 20))
        self.lbl_track_title = ctk.CTkLabel(self.info_frame, text="En attente...", font=("Arial", 24, "bold"), text_color="white")
        self.lbl_track_title.pack()
        self.lbl_artist = ctk.CTkLabel(self.info_frame, text="AI Generator Model v1.0", font=("Arial", 14), text_color="#b3b3b3")
        self.lbl_artist.pack()

        # Status
        self.dynamic_status_container = ctk.CTkFrame(self, fg_color="transparent")
        self.dynamic_status_container.grid(row=3, column=1, pady=(0, 30))
        self.progress_bar = ctk.CTkProgressBar(self.dynamic_status_container, width=500, height=6, progress_color="#1db954")
        self.progress_bar.set(0)

        # Player
        self.audio_controls = ctk.CTkFrame(self.dynamic_status_container, fg_color="transparent")
        self.btn_play = ctk.CTkButton(self.audio_controls, text="▶", width=50, height=50, corner_radius=25, 
                                      fg_color="white", text_color="black", font=("Arial", 20), hover_color="#dddddd", 
                                      command=self.toggle_play_audio)
        self.btn_play.grid(row=0, column=0, padx=(0, 20))
        
        self.slider_audio = ctk.CTkSlider(self.audio_controls, width=350, height=20, button_color="white", 
                                          progress_color="#1db954", from_=0, to=10)
        self.slider_audio.set(0)
        self.slider_audio.grid(row=0, column=1, padx=(0, 10))
        self.lbl_time = ctk.CTkLabel(self.audio_controls, text="0:00 / 0:10", font=("Arial", 12), text_color="#aaaaaa")
        self.lbl_time.grid(row=0, column=2)

        # Input
        self.input_container = ctk.CTkFrame(self, fg_color="transparent")
        self.input_container.grid(row=4, column=1, pady=(0, 30))
        self.entry_text = ctk.CTkEntry(self.input_container, width=400, height=45, placeholder_text="Décris l'émotion ou le son...", 
                                       corner_radius=25, border_width=0, fg_color="#2a2a2a", text_color="white", font=("Arial", 14))
        self.entry_text.pack(side="left", padx=(0, 15))
        self.btn_run = ctk.CTkButton(self.input_container, text="GÉNÉRER", height=45, width=140, corner_radius=25, 
                                     fg_color="#1db954", hover_color="#1ed760", font=("Arial", 13, "bold"), text_color="black", 
                                     command=self.on_generate_click)
        self.btn_run.pack(side="left")

        # Console 
        self.frame_details = ctk.CTkFrame(self, fg_color="#000000", height=200, corner_radius=0)
        self.setup_admin_panel()


    # --- LOGIQUE ADMIN ---
    def setup_admin_panel(self):
        ctk.CTkLabel(self.frame_details, text="ENGINEERING CONSOLE", font=("Courier", 12, "bold"), text_color="#1db954").pack(anchor="w", padx=20, pady=10)
        self.entry_prompt_debug = ctk.CTkEntry(self.frame_details, fg_color="#111111", border_color="#333333", text_color="#00ff00", font=("Courier", 12))
        self.entry_prompt_debug.pack(fill="x", padx=20, pady=5)
        self.textbox_logs = ctk.CTkTextbox(self.frame_details, fg_color="#111111", text_color="#dddddd", font=("Courier", 11), height=100)
        self.textbox_logs.pack(fill="x", padx=20, pady=(0,20))

    def toggle_admin_panel(self):
        if self.frame_details.winfo_viewable():
            self.frame_details.grid_forget()
        else:
            if self.is_admin_unlocked:
                self.frame_details.grid(row=5, column=1, sticky="ew") 
            else:
                self.open_login_window()

    def open_login_window(self):
        login = ctk.CTkToplevel(self)
        login.title("Admin Access")
        login.geometry("300x180")
        login.attributes("-topmost", True)
        login.configure(fg_color="#1e1e1e")
        ctk.CTkLabel(login, text="Dev Access", font=("Arial", 14, "bold")).pack(pady=10)
        u_entry = ctk.CTkEntry(login, placeholder_text="User"); u_entry.pack(pady=5)
        p_entry = ctk.CTkEntry(login, placeholder_text="Pass", show="*"); p_entry.pack(pady=5)
        def check():
            if u_entry.get() == "admin" and p_entry.get() == "admin":
                self.is_admin_unlocked = True
                self.frame_details.grid(row=5, column=1, sticky="ew")
                login.destroy()
        ctk.CTkButton(login, text="Unlock", fg_color="#333", command=check).pack(pady=10)

    # --- LOGIQUE VISUELLE ---
    def toggle_view_mode(self):
        if self.view_mode == "COVER":
            self.cover_frame.grid_forget()
            self.graph_frame.grid(row=0, column=0, sticky="nsew")
            self.view_mode = "GRAPH"
            self.btn_view.configure(text="Pochette", fg_color="#555555")
        else:
            self.graph_frame.grid_forget()
            self.cover_frame.grid(row=0, column=0)
            self.view_mode = "COVER"
            self.btn_view.configure(text="Signal", fg_color="#333333")

    def update_plots(self, emotion_label="neutre"):
        self.axs[0].clear(); self.axs[1].clear()
        for ax in self.axs:
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
        
        fs = 44100; duration = 0.05
        t = np.linspace(0, duration, int(fs*duration))
        freq_base = 440
        if "calme" in emotion_label or "triste" in emotion_label:
            freq_base = 220
            y = 0.8 * np.sin(2 * np.pi * freq_base * t)
        elif "colere" in emotion_label or "peur" in emotion_label:
            freq_base = 150
            y = 0.6 * np.sign(np.sin(2 * np.pi * freq_base * t)) + 0.3 * np.random.normal(0, 1, len(t))
        else:
            freq_base = 880
            y = 0.5 * np.sin(2 * np.pi * freq_base * t) + 0.3 * np.sin(2 * np.pi * (freq_base*1.5) * t)

        self.axs[0].plot(t*1000, y, color='#1db954', linewidth=1.5)
        self.axs[0].set_title("Amplitude (Time Domain)", color='white', fontsize=10, pad=10)
        self.axs[0].set_xlabel("Time (ms)", color='#888888', fontsize=9)
        self.axs[0].set_ylim(-1.5, 1.5)

        t_long = np.linspace(0, 1, fs)
        if "calme" in emotion_label: y_long = 0.8 * np.sin(2 * np.pi * freq_base * t_long)
        else: y_long = y
        N = len(y); yf = np.fft.fft(y); xf = np.fft.fftfreq(N, 1/fs)[:N//2]
        mag = 2.0/N * np.abs(yf[0:N//2])

        self.axs[1].fill_between(xf, mag, color='#1db954', alpha=0.3)
        self.axs[1].plot(xf, mag, color='white', linewidth=1)
        self.axs[1].set_title("Spectrum (Frequency Domain)", color='white', fontsize=10, pad=10)
        self.axs[1].set_xlabel("Frequency (Hz)", color='#888888', fontsize=9)
        self.axs[1].set_xlim(0, 2000); self.axs[1].set_yticks([])
        self.canvas.draw()

    # --- LOGIQUE GENERATION ---
    def on_generate_click(self):
        text = self.entry_text.get()
        if not text: return
        self.btn_run.configure(state="disabled", text="CALCUL...", fg_color="#555555")
        self.lbl_track_title.configure(text=text, text_color="#aaaaaa")
        self.audio_controls.pack_forget(); self.progress_bar.pack(); self.progress_bar.set(0)
        threading.Thread(target=self.run_process, args=(text,)).start()

    def run_process(self, text):
        for i in range(20): time.sleep(0.05); self.progress_bar.set((i+1)/20)
        prompt, img, audio = self.ai.process_pipeline(text)
        self.after(0, self.show_results, prompt, img, audio, text)

    def show_results(self, prompt, img, audio, original_text):

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(400, 400))
        self.lbl_image.configure(image=ctk_img, text="")
        self.lbl_track_title.configure(text=original_text, text_color="white")
        self.entry_prompt_debug.delete(0, "end"); self.entry_prompt_debug.insert(0, f"PROMPT> {prompt}")
        self.textbox_logs.delete("0.0", "end"); self.textbox_logs.insert("end", f"[SUCCESS] Audio generated at {audio}\n")
        
        emo = "neutre"
        if "[emotion:" in prompt: emo = prompt.split("[emotion:")[1].split("]")[0]
        self.update_plots(emo)

        self.progress_bar.pack_forget(); self.audio_controls.pack()
        self.btn_run.configure(state="normal", text="GÉNÉRER", fg_color="#1db954")

        self.add_to_history(original_text, prompt, emo, img, audio)

    # --- GESTION HISTORIQUE ---
    def add_to_history(self, text, prompt, emotion, img, audio):
        """Crée un bouton dans la sidebar pour rappeler cette session"""
        self.lbl_empty_hist.pack_forget() 
        
        # Capture de l'heure
        now = datetime.datetime.now().strftime("%H:%M")
        
        # Sauvegarde des données
        session_data = {
            "text": text, "prompt": prompt, "emotion": emotion,
            "img": img, "audio": audio, "time": now
        }
        self.session_history.append(session_data)
        
        # Création du bouton visuel
        idx = len(self.session_history) - 1
        
        btn_text = f"{now} | {emotion.upper()}\n{text[:25]}..."
        btn = ctk.CTkButton(self.scroll_history, text=btn_text, anchor="w",
                            fg_color="#2b2b2b", hover_color="#333333",
                            height=50, width=220, font=("Arial", 11),
                            command=lambda i=idx: self.restore_session(i))
        btn.pack(pady=5, padx=5)

    def restore_session(self, index):
        """Recharge une ancienne session quand on clique dessus"""
        data = self.session_history[index]
        
        self.lbl_track_title.configure(text=data["text"], text_color="white")
        self.entry_text.delete(0, "end"); self.entry_text.insert(0, data["text"])
        
        ctk_img = ctk.CTkImage(light_image=data["img"], dark_image=data["img"], size=(400, 400))
        self.lbl_image.configure(image=ctk_img, text="")
        
        self.entry_prompt_debug.delete(0, "end"); self.entry_prompt_debug.insert(0, f"PROMPT> {data['prompt']}")
        self.textbox_logs.insert("end", f"[RESTORE] Loaded session from {data['time']}\n")
        
        self.update_plots(data["emotion"])
        
        self.progress_bar.pack_forget()
        self.audio_controls.pack()

    # --- PLAYER AUDIO ---
    def toggle_play_audio(self):
        if not self.is_playing:
            self.is_playing = True; self.btn_play.configure(text="❚❚")
            self.update_slider_loop()
        else:
            self.is_playing = False; self.btn_play.configure(text="▶")

    def update_slider_loop(self):
        if self.is_playing:
            current_val = self.slider_audio.get()
            if current_val < 10:
                new_val = current_val + 0.1
                self.slider_audio.set(new_val)
                seconds = int(new_val)
                self.lbl_time.configure(text=f"0:{seconds:02d} / 0:10")
                self.after(100, self.update_slider_loop)
            else:
                self.is_playing = False; self.btn_play.configure(text="▶")
                self.slider_audio.set(0); self.lbl_time.configure(text="0:00 / 0:10")