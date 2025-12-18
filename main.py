import customtkinter as ctk
from ui_interface import AppInterface

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    app = AppInterface()
    app.mainloop()