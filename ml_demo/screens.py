import os

import ml_components.models.neural_network as nn
import pandas as pd

from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from libs.garden.graph import MeshLinePlot
from libs.garden.touchgraph import TouchGraph

from ml_demo.dialogs import LoadDialog, SaveDialog, GraphDialog


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

    def show_load_training_data(self):
        content = LoadDialog(load=self.load_training_data, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Data", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load_predict_data(self):
        content = LoadDialog(load=self.load_predict_data, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Data", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load_model(self):
        content = LoadDialog(load=self.load_model, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Model", content=content, size_hint=(0.9, 0.9))

    def show_save_model(self):
        content = SaveDialog(save=self.save_model, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save Model", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load_training_data(self, path, filename):
        raw_data = pd.read_csv(filename[0]).as_matrix()
        self.data = self.split_data(raw_data)

        self.train_button.disabled = False
        self.dismiss_popup()

    def load_predict_data(self, path, filename):
        self.data = pd.read_csv(filename[0]).as_matrix()

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
        self.accuracy = None

    def train(self):
        print("training")
        lam = self.lam.value
        alpha = self.alpha.value
        epochs = self.epochs.value
        adaptive = self.adaptive.value
        dec_amount = self.dec_amount.value
        classes = self.classes.value
        hidden_layer_size = self.hidden_layer_size.value

        self.network = nn.NeuralNetwork(X=self.data['X'], y=self.data['y'], classes=int(classes),
                                        hidden_layer_size=int(hidden_layer_size), lam=lam, activation_func='sigmoid')

        self.cost, self.costs, self.model = self.network.train(alpha=alpha, max_epochs=int(epochs), adaptive=adaptive,
                                                               dec_amount=dec_amount)

        self.training_graph_button.disabled = False

    def display_training_graph(self):
        points = []

        for i, cost in enumerate(self.costs):
            points.append(i)
            points.append(cost)

        content = GraphDialog(cancel=self.dismiss_popup)
        content.graph.points = points
        content.max_y = max(self.costs) + max(self.costs) * 0.01
        content.min_y = min(self.costs) - min(self.costs) * 0.01
        content.x_ticks = [x for x, _ in enumerate(self.costs)]

        self._popup = Popup(title="Training Graph", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def predict(self):
        self.network.predict(self.data)


class DecisionTreeScreen(ModelScreen):
    def train(self):
        pass

    def predict(self):
        pass

    def load_model(self, path, filename):
        #TODO: custom loading for decision tree model
        pass
