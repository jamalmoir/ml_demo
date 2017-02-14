import os

from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen

from ml_demo.dialogs import LoadDialog, SaveDialog


class MainScreen(Screen):
    pass


class ModelScreen(Screen):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def __init__(self, **kw):
        super().__init__(**kw)

        self.model = None
        self.data = None

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            loaded = stream.read()

        self.dismiss_popup()

        return loaded

    def save(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text_input.text)

        self.dismiss_popup()

    def load_model(self):
        self.model = self.show_load()

    def load_data(self):
        self.data = self.show_load()

    def enter_data(self):
        pass

    def save_model(self):
        self.show_save()

    def clear(self):
        self.model = None
        self.data = None


class NeuralNetworkScreen(ModelScreen):
    def train(self):
        pass

    def display_training_graph(self):
        pass

    def predict(self):
        pass


class DecisionTreeScreen(ModelScreen):
    def train(self):
        pass

    def predict(self):
        pass