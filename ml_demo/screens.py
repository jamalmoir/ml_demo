import os
from collections import namedtuple

import ml_components.models.neural_network as nn
import pandas as pd

from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.graphics import Color, Ellipse, Line
from random import random

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
        self.draw_network()

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

    def get_bottom_padding(self, d, num_nodes, base_y, y_offset):
        return base_y + self.nn_graphic.size[1] / 2 - ((num_nodes * d + (num_nodes - 1) * y_offset) / 2)

    def draw_network(self):
        GraphNode = namedtuple('GraphNode', ['x', 'y'])
        GraphLine = namedtuple('GraphLine', ['x1', 'y1', 'x2', 'y2'])
        graph_nodes = [[], [], []]
        graph_lines = [[], []]

        color = (0.22, 0.22, 0.22, 1)

        base_x = self.nn_graphic.pos[0]
        base_y = self.nn_graphic.pos[1]

        input = self.network.input_layer_size
        hidden = self.network.hidden_layer_size if self.network.hidden_layer_size <= 50 \
            else int(self.network.hidden_layer_size / 10)
        output = self.network.label_count

        d = (self.nn_graphic.size[1] * 0.75) / max([input, hidden, output])

        x_offset = (self.nn_graphic.size[0] * 0.5) - d / 2
        y_offset = (self.nn_graphic.size[1] - (max([input, hidden, output]) * d)) / max([input, hidden, output])

        # Calculate input layer node positions.
        for i in range(input):
            # Padding below the set of node to centre them.
            bottom_padding = self.get_bottom_padding(d=d, num_nodes=input, base_y=base_y, y_offset=y_offset)
            y = bottom_padding + (y_offset * i) + (d * i)

            graph_nodes[0].append(GraphNode(x=base_x, y=y))

        # Calculate hidden layer node positions.
        for i in range(hidden):
            # Padding below the set of node to centre them.
            bottom_padding = self.get_bottom_padding(d=d, num_nodes=hidden, base_y=base_y, y_offset=y_offset)
            y = bottom_padding + (y_offset * i) + (d * i)
            x = base_x + x_offset

            graph_nodes[1].append(GraphNode(x=x, y=y))

        # Calculate output layer node positions.
        for i in range(output):
            # Padding below the set of node to centre them.
            bottom_padding = self.get_bottom_padding(d=d, num_nodes=output, base_y=base_y, y_offset=y_offset)
            x = base_x + 2 * x_offset
            y = bottom_padding + (y_offset * i) + (d * i)

            graph_nodes[2].append(GraphNode(x=x, y=y))

        # Calculate input to hidden line x and ys.
        for node0 in graph_nodes[0]:
            for node1 in graph_nodes[1]:
                graph_lines[0].append(GraphLine(x1=node0.x, y1=node0.y + (d / 2), x2=node1.x, y2=node1.y + (d / 2)))

        # Calculate hidden to output line x and ys.
        for node1 in graph_nodes[1]:
            for node2 in graph_nodes[2]:
                graph_lines[1].append(GraphLine(x1=node1.x, y1=node1.y + (d / 2), x2=node2.x, y2=node2.y + (d / 2)))

        with self.nn_graphic.canvas:
            Color(*color)

            # Draw nodes.
            for layer in graph_nodes:
                for node in layer:
                    Ellipse(pos=(node.x, node.y), size=(d, d))

            # Draw lines.
            for layer in graph_lines:
                for line in layer:
                    Line(points=[line.x1, line.y1, line.x2, line.y2])


class DecisionTreeScreen(ModelScreen):
    def train(self):
        pass

    def predict(self):
        pass

    def load_model(self, path, filename):
        #TODO: custom loading for decision tree model
        pass
