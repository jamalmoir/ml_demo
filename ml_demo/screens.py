import os
import pickle
import io

from collections import namedtuple

from ml_demo.text import text_eng
from ml_demo.dialogs import LoadDialog, SaveDialog, GraphDialog
from ml_demo.utils.data_tools import split_data
from ml_demo.utils.graph_tools import get_bottom_padding, TreeTraverse

from ml_components.models import neural_network, decision_tree

import numpy as np
import pandas as pd
import graphviz as gv

from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.core.image import Image
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.clock import Clock

from libs.garden.xpopup import XError, XNotification, XMessage


class MainScreen(Screen):
    pass


class ModelScreen(Screen):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def __init__(self, **kw):
        super().__init__(**kw)

        self.model = None
        self.training_data = None
        self.predict_data = None
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
        self._popup.open()

    def show_save_model(self):
        content = SaveDialog(save=self.save_model, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save Model", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load_training_data(self, path, filename):
        try:
            raw_data = pd.read_csv(filename[0]).as_matrix()
            self.training_data = split_data(raw_data, train_test=self.train_test)

            self.train_button.disabled = False
            self.dismiss_popup()
        except Exception as e:
            self.dismiss_popup()
            XError(text='Error loading training data: {}'.format(e))

    def load_predict_data(self, path, filename):
        try:
            self.predict_data = pd.read_csv(filename[0]).as_matrix()
            self.dismiss_popup()

            if self.model:
                self.predict_button.disabled = False
        except Exception as e:
            self.dismiss_popup()
            XError(text='Error loading predicting data!: {}'.format(e))

    def load_model(self, path, filename):
        try:
            with open(os.path.join(path, filename[0]), 'rb') as stream:
                self.model = pickle.load(stream)

            if self.predict_data:
                self.predict_button.disabled = False

            self.dismiss_popup()
        except Exception as e:
            self.dismiss_popup()
            XError(text='Error loading model!: {}'.format(e))

    def save_model(self, path, filename):
        try:
            with open(os.path.join(path, filename), 'wb') as stream:
                pickle.dump(self.model, stream)

            self.dismiss_popup()
        except Exception as e:
            self.dismiss_popup()
            XError(text='Error saving model!: {}'.format(e))

    def clear(self):
        self.model = None
        self.training_data = None
        self.predict_data = None
        self.predict_button.disabled = True
        self.train_button.disabled = False

    def begin_training(self):
        self.loading_pop = XNotification(title='Training...', text='Your model is training, please wait...')
        self.loading_pop.open()
        Clock.schedule_once(self.train, 0)


class NeuralNetworkScreen(ModelScreen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.network = None
        self.cost = 0
        self.costs = []
        self.accuracy = 0

    def clear(self):
        super().clear()

        self.nn_graphic.canvas.clear()
        self.network = None
        self.cost = 0
        self.costs = []
        self.accuracy = 0
        self.train_button.disabled = True
        self.predict_button.disabled = True
        self.training_graph_button.disabled = True

    def load_model(self, path, filename):
        super().load_model(path, filename)

        self.network = neural_network.NeuralNetwork(model=self.model)
        self.draw_network()

    def train(self, *args):
        lam = self.lam.value
        alpha = self.alpha.value
        epochs = self.epochs.value
        adaptive = self.adaptive.value
        dec_amount = self.dec_amount.value
        hidden_layer_size = self.hidden_layer_size.value

        try:
            self.network = neural_network.NeuralNetwork(hidden_layer_size=int(hidden_layer_size),
                                                        activation_func='sigmoid')

            self.cost, self.costs, self.model = self.network.train(X=self.training_data['X'],
                                                                   y=self.training_data['y'],
                                                                   alpha=alpha,
                                                                   max_epochs=int(epochs),
                                                                   lam=lam,
                                                                   adaptive=adaptive,
                                                                   dec_amount=dec_amount)
            self.loading_pop.dismiss()

            if self.train_test:
                accuracy = self.get_accuracy()

                XMessage(text='Training complete! Your Model\'s accuracy is {acc:.2f}%.'.format(acc=accuracy),
                         title='Your Model Has Been Trained!')
            else:
                XMessage(text='Training complete!', title='Your Model Has Been Trained!')

            if self.predict_data:
                self.predict_button.disabled = False

            self.training_graph_button.disabled = False
            self.draw_network()
        except Exception as e:
            self.loading_pop.dismiss()
            XError(text='Error training model!: {}'.format(e))

    def get_accuracy(self):
        pred = self.network.predict(self.training_data['X_test'])
        accuracy = np.sum(self.training_data['y_test'] == pred, axis=0) / self.training_data['X_test'].shape[0] * 100

        return accuracy

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
        try:
            prediction = self.network.predict(self.predict_data)

            XMessage(text='Your model predicts {pred}'.format(pred=prediction), title='Prediction')
        except Exception as e:
            XError(text='Error making prediction!: {}'.format(e))

    def draw_network(self):
        self.nn_graphic.canvas.clear()  # Make sure canvas is empty.

        GraphNode = namedtuple('GraphNode', ['x', 'y'])
        GraphLine = namedtuple('GraphLine', ['x1', 'y1', 'x2', 'y2'])
        graph_nodes = [[], [], []]
        graph_lines = [[], []]

        # Get the number of nodes in each layer and limit them so the network graph is legible.
        input_count = self.network._input_layer_size if self.network._input_layer_size <= 50 \
            else int(self.network._input_layer_size / 10)
        hidden_count = self.network._hidden_layer_size if self.network._hidden_layer_size <= 50 \
            else int(self.network._hidden_layer_size / 10)
        output_count = self.network._output_layer_size if self.network._output_layer_size <= 50 \
            else int(self.network._output_layer_size / 10)

        d = (self.nn_graphic.size[1] * 0.75) / max([input_count, hidden_count, output_count])  # Node diameter.
        color = (0.22, 0.22, 0.22, 1)  # Graph color.
        base_x = self.nn_graphic.pos[0]  # The x coord of the bottom left corner of the graphic area.
        base_y = self.nn_graphic.pos[1]  # The y coord of the bottom left corner of the graphic area.
        x_offset = (self.nn_graphic.size[0] * 0.5) - d / 2  # Space between nodes on the x axis.
        y_offset = (self.nn_graphic.size[1] - (max([input_count, hidden_count, output_count]) * d)) / \
            max([input_count, hidden_count, output_count])  # Space between nodes on the y axis.

        # Calculate input layer node positions.
        for i in range(input_count):
            # Padding below the set of node to centre them.
            bottom_padding = get_bottom_padding(height=self.nn_graphic.size[1],
                                                d=d,
                                                num_nodes=input_count,
                                                base_y=base_y,
                                                y_offset=y_offset)

            y = bottom_padding + (y_offset * i) + (d * i)

            graph_nodes[0].append(GraphNode(x=base_x, y=y))

        # Calculate hidden layer node positions.
        for i in range(hidden_count):
            # Padding below the set of node to centre them.
            bottom_padding = get_bottom_padding(height=self.nn_graphic.size[1],
                                                d=d,
                                                num_nodes=hidden_count,
                                                base_y=base_y,
                                                y_offset=y_offset)

            y = bottom_padding + (y_offset * i) + (d * i)
            x = base_x + x_offset

            graph_nodes[1].append(GraphNode(x=x, y=y))

        # Calculate output layer node positions.
        for i in range(output_count):
            # Padding below the set of node to centre them.
            bottom_padding = get_bottom_padding(height=self.nn_graphic.size[1],
                                                d=d,
                                                num_nodes=output_count,
                                                base_y=base_y,
                                                y_offset=y_offset)

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

        # TODO: Fix resizing.
        self.get_root_window().bind(on_resize=lambda a, b, c: self.draw_network())
        self.get_root_window().bind(on_maximize=lambda a, b, c: self.draw_network())
        self.get_root_window().bind(on_minimize=lambda a, b, c: self.draw_network())


class DecisionTreeScreen(ModelScreen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.decision_tree = None
        self.accuracy = 0

    def clear(self):
        super().clear()

        self.dt_graphic.canvas.clear()
        self.decision_tree = None
        self.accuracy = 0
        self.train_button.disabled = True
        self.predict_button.disabled = True

    def load_model(self, path, filename):
        super().load_model(path, filename)

        self.decision_tree = decision_tree.DecisionTree(model=self.model)
        self.draw_tree()

    def train(self, *args):
        # try:
            self.decision_tree = decision_tree.DecisionTree()
            self.model = self.decision_tree.train(X=self.training_data['X'], y=self.training_data['y'])
            self.draw_tree()

            self.loading_pop.dismiss()

            if self.train_test:
                accuracy = self.get_accuracy()

                XMessage(text='Training complete! Your Model\'s accuracy is {acc:.2f}%.'.format(acc=int(accuracy)),
                         title='Your Model Has Been Trained!')
            else:
                XMessage(text='Training complete!', title='Your Model Has Been Trained!')

            if self.predict_data:
                self.predict_button.disabled = False

        # except Exception as e:
            # self.loading_pop.dismiss()
            # XError(text='Error training model!: {}'.format(e))

    def predict(self):
        try:
            prediction = self.decision_tree.predict(self.predict_data)

            XMessage(text='Your model predicts {pred}'.format(pred=prediction), title='Prediction')
        except Exception as e:
            XError(text='Error making prediction!: {}'.format(e))

    def get_accuracy(self):
        pred = self.decision_tree.predict(self.training_data['X_test'])
        accuracy = np.sum(self.training_data['y_test'] == pred, axis=0) / self.training_data['X_test'].shape[0] * 100

        return accuracy

    def draw_tree(self):
        self.dt_graphic.canvas.clear()  # Make sure canvas is empty.

        tt = TreeTraverse()
        nodes, edges = tt.get_nodes_and_edges(model=self.model['structure'])

        graph = gv.Digraph(format='jpg')
        graph.body.append('style=filled')
        # graph.body.append('color=0.96 0.96 0.96')
        # graph.body.append('color=#f5f5f5')
        graph.body.append('color=blue')

        for node in nodes:
            label = node.output if node.output else node.attribute
            graph.node(name=str(node.num), label=str(label))

        for edge in edges:
            graph.edge(str(edge.a), str(edge.b))

        graph_img = io.BytesIO(graph.pipe())

        with self.dt_graphic.canvas:
            self.dt_graphic.bg = Rectangle(texture=Image(graph_img, ext='jpg').texture)
            self.dt_graphic.bg.pos = self.dt_graphic.pos
            self.dt_graphic.bg.size = self.dt_graphic.size

        self.get_root_window().bind(on_resize=lambda a, b, c: self.draw_tree())
        self.get_root_window().bind(on_maximize=lambda a, b, c: self.draw_tree())
        self.get_root_window().bind(on_minimize=lambda a, b, c: self.draw_tree())


class HelpScreen(Screen):
    pass


class NeuralNetworkHelpScreen(HelpScreen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.help_title.text = 'Neural Network Help'
        self.help_text.text = text_eng['nn_help_text']


class DecisionTreeHelpScreen(HelpScreen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.help_title.text = 'Decision Tree Help'
        self.help_text.text = text_eng['dt_help_text']
