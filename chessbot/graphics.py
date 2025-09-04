import pygame
from pygame.locals import *
from game import GameManager

TILE_SIZE = 80
BOARD_SIZE = TILE_SIZE * 8

pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Chess")

piece_images = {
    "P": pygame.image.load("assets/white_pawn.png"),
    "p": pygame.image.load("assets/black_pawn.png"),
    "n": pygame.image.load("assets/black_knight.png"),
    "N": pygame.image.load("assets/white_knight.png"),
    "B": pygame.image.load("assets/white_bishop.png"),
    "b": pygame.image.load("assets/black_bishop.png"),
    "R": pygame.image.load("assets/white_rook.png"),
    "r": pygame.image.load("assets/black_rook.png"),
    "K": pygame.image.load("assets/white_king.png"),
    "k": pygame.image.load("assets/black_king.png"),
    "Q": pygame.image.load("assets/white_queen.png"),
    "q": pygame.image.load("assets/black_queen.png"),
}

def draw_board(screen, board, selected):
    colors = [(240, 217, 181), (181, 136, 99)]  # light/dark squares
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            rect = pygame.Rect(col*TILE_SIZE, row*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, color, rect)

            # Highlight selected square
            if selected == (row, col):
                pygame.draw.rect(screen, (0, 255, 0), rect, 3)

            piece = board.getPieceAt((row, col))
            if piece:
                piece_img = piece_images[str(piece)]
                piece_rect = piece_img.get_rect(center = rect.center)
                screen.blit(piece_img, piece_rect)



game = GameManager()  # this has .board inside
clock = pygame.time.Clock()
selected_square = None  # track clicks

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // TILE_SIZE, x // TILE_SIZE

            if selected_square is None:
                # First click: select a piece
                piece = game.board.getPieceAt((row, col))
                if piece and piece.colour == game.board.turn:
                    selected_square = (row, col)
            else:
                # Second click: try to move
                start_pos = selected_square
                end_pos = (row, col)
                game.board.movePiece(start_pos, end_pos, switch_turn=True)
                selected_square = None

    # --- Drawing ---
    draw_board(screen, game.board, selected_square)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()