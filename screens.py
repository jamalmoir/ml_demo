from kivy.uix.screenmanager import Screen


class MainScreen(Screen):
    pass


class ModelScreen(Screen):
    def load_model(self):
        pass

    def load_data(self):
        pass

    def enter_data(self):
        pass

    def save_model(self):
        pass

    def clear(self):
        pass


class NeuralNetworkScreen(ModelScreen):
    def train(self):
        pass

    def display_training_graph(self):
        pass

    def predict(self):
        pass


class DecisionTreeScreen(Screen):
    def train(self):
        pass

    def predict(self):
        pass