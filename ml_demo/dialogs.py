from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, ListProperty


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)


class GraphDialog(FloatLayout):
    cancel = ObjectProperty(None)
