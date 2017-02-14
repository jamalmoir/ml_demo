from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config

Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'resizable', False)

Builder.load_file('mlgui.kv')


class MainScreen(Screen):
    pass


class NeuralNetworkScreen(Screen):
    pass


class DecisionTreeScreen(Screen):
    pass

screen_manager = ScreenManager()

screen_manager.add_widget(MainScreen(name='main_screen'))
screen_manager.add_widget(NeuralNetworkScreen(name='nn_screen'))
screen_manager.add_widget(DecisionTreeScreen(name='dt_screen'))


class MLGUIApp(App):

    def build(self):
        return screen_manager

mlgui = MLGUIApp()
mlgui.run()
