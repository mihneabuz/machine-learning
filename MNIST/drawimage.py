import arcade

IMAGE_SIZE = 28
SCALE_FACTOR = 24
SCREEN_WIDTH = IMAGE_SIZE * SCALE_FACTOR
SCREEN_HEIGHT = IMAGE_SIZE * SCALE_FACTOR
SCREEN_TITLE = "Draw a digit!"
BRUSH = [[0, 32, 0], [32, 96, 32], [0, 32, 0]]


image = []

class DrawDigit(arcade.Window):

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.tiles = [[0 for j in range(28)] for i in range(28)]
        self.drawing = False
        self.brush_size = 3
        self.brush = BRUSH
    def on_draw(self):
        for i in range(28):
            for j in range(28):
                color = [self.tiles[i][j]] * 3
                arcade.draw_rectangle_filled((j + 1/2) * SCALE_FACTOR,
                                             SCREEN_HEIGHT - (i + 1/2) * SCALE_FACTOR,
                                             SCALE_FACTOR, SCALE_FACTOR, color)

    def on_mouse_release(self, x, y, button, modifiers):
        self.drawing = False

    def on_mouse_press(self, x, y, button, modifiers):
        self.drawing = True
        j = int(x / SCALE_FACTOR)
        i = int((SCREEN_HEIGHT - y) / SCALE_FACTOR)
        self.tiles[i][j] = 255

    def on_mouse_motion(self, x, y, dx, dy):
        if (self.drawing):
            x_corner = int(x / SCALE_FACTOR - (self.brush_size - 1) / 2)
            y_corner = int((SCREEN_HEIGHT - y) / SCALE_FACTOR - (self.brush_size - 1) / 2)
            for i in range(self.brush_size):
                for j in range(self.brush_size):
                    if (y_corner + i >= 0 and y_corner + i < 28 and
                        x_corner + j >= 0 and x_corner + j < 28):
                        self.tiles[y_corner + i][x_corner + j] += self.brush[i][j]
                        if (self.tiles[y_corner + i][x_corner + j] > 255):
                            self.tiles[y_corner + i][x_corner + j] = 255

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ENTER:
            global image
            image = self.tiles
            arcade.window_commands.close_window()

def drawImage():
    DrawDigit()
    arcade.run()
    global image
    return image

if (__name__ == "__main__"):
    drawImage()

