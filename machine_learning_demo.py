from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen


Builder.load_file('mlgui.kv')

class MainScreen(Screen):
    pass

screen_manager = ScreenManager()

screen_manager.add_widget(MainScreen(name='main'))


class MLGUIApp(App):

    def build(self):
        return screen_manager

mlgui = MLGUIApp()
mlgui.run()
