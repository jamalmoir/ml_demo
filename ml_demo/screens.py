import os

import ml_components.models.neural_network as nn
import pandas as pd

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
        self.loaded_file = None

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load_data(self):
        content = LoadDialog(load=self.load_data, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Data", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load_model(self):
        content = LoadDialog(load=self.load_model, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Model", content=content, size_hint=(0.9, 0.9))

    def show_save_model(self):
        content = SaveDialog(save=self.save_model, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save Model", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load_data(self, path, filename):
        raw_data = pd.read_csv(filename[0]).as_matrix()
        self.data = self.split_data(raw_data)

        self.dismiss_popup()

    def load_model(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            loaded_file = stream.read()
            self.model = pd.read_csv(loaded_file).as_matrix()

    def save_model(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.model)

        self.dismiss_popup()

    def clear(self):
        self.model = None
        self.data = None

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]

        return {'X': X, 'y': y}


class NeuralNetworkScreen(ModelScreen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.network = None
        self.cost = None
        self.costs = None
        self.model = None

    def train(self):
        print("training")
        lam = self.lam.value
        alpha = self.alpha.value
        epochs = self.epochs.value
        adaptive = self.adaptive.value
        dec_amount = self.dec_amount.value
        classes = self.classes.value
        hidden_layer_size = self.hidden_layer_size.value

        self.network = nn.NeuralNetwork(X=self.data['X'], y=self.data['y'], classes=classes,
                                        hidden_layer_size=hidden_layer_size, lam=lam, activation_func='sigmoid')

        self.cost, self.costs, self.model = self.network.train(alpha=alpha, max_epochs=epochs, adaptive=adaptive,
                                                               dec_amount=dec_amount)

        print(self.cost)

    def display_training_graph(self):
        pass

    def predict(self):
        pass


class DecisionTreeScreen(ModelScreen):
    def train(self):
        pass

    def predict(self):
        pass
