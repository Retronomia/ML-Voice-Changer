import keyboard
import os

class HotkeyManager:
    EXIT_KEY = 'q'
    TOGGLE_MUTE = 'v'

    def __init__(self):
        self.muted = True

        self.setup_hotkeys()

    def setup_hotkeys(self):

        self.add_hotkey([self.EXIT_KEY], self.exit_key_pressed)
        self.add_hotkey([self.TOGGLE_MUTE], self.toggle_mute)

    def exit_key_pressed(self):
        print(f"Exit hotkey {self.EXIT_KEY} pressed, exiting program")
        os._exit(1)

    def toggle_mute(self):
        self.muted = not self.muted

        if self.muted:
            print("Now Muted")
        else:
            print("No Longer Muted")

    def add_hotkey(self, hotkeys: [str], func):
        # Combines hotkeys in special format
        # https://github.com/boppreh/keyboard#keyboardadd_hotkeyhotkey-callback-args-suppressfalse-timeout1-trigger_on_releasefalse
        hotkey = '+'.join(hotkeys)
        keyboard.add_hotkey('+'.join(hotkey), func)

    def is_muted(self):
        return self.muted
