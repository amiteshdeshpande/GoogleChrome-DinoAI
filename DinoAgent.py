# The file that initializes a dino agent and provides methods for
# actions that can be performed

from GameModule import GameModule

class DinoAgent:
    # The constructor to initialize the dino agent
    # We perform jump action to start the game
    def __init__(self,game):
        self._game = game;
        self.jump();

    # Function to see if the game is running
    def is_running(self):
        return self._game.get_playing()

    # Function to see if the game is over
    def is_crashed(self):
        return self._game.get_crashed()

    # Function to make the Dino jump
    def jump(self):
        self._game.press_up()

    # Function to make the Dino duck
    def duck(self):
        self._game.press_down()
