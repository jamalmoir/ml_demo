import os

from kivy.factory import Factory
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivy.config import Config
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

from screens import MainScreen, NeuralNetworkScreen, DecisionTreeScreen
from dialogs import LoadDialog, SaveDialog

Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'resizable', False)

Builder.load_file('mlgui.kv')

screen_manager = ScreenManager()

screen_manager.add_widget(MainScreen(name='main_screen'))
screen_manager.add_widget(NeuralNetworkScreen(name='nn_screen'))
screen_manager.add_widget(DecisionTreeScreen(name='dt_screen'))


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

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


class MLGUIApp(App):

    def build(self):
        return screen_manager


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

mlgui = MLGUIApp()
mlgui.run()
