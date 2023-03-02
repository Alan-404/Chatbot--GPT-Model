from kivy.app import App
from kivy.uix.label import Label
class MainApp(App):
    def build(self):
        self.lb = Label(text="Circle")