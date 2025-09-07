import pygame, sys
from pygame.locals import *
from game import GameManager
from button import Button

TILE_SIZE = 80
BOARD_SIZE = TILE_SIZE * 8

pygame.init()
SCREEN = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
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

def draw_board(board, selected):
    colors = [(240, 217, 181), (181, 136, 99)]  # light/dark squares
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            rect = pygame.Rect(col*TILE_SIZE, row*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(SCREEN, color, rect)

            # Highlight selected square
            if selected == (row, col):
                pygame.draw.rect(SCREEN, (255, 255, 255), rect, 3)

            piece = board.getPieceAt((row, col))
            if piece:
                piece_img = piece_images[str(piece)]
                piece_rect = piece_img.get_rect(center = rect.center)
                SCREEN.blit(piece_img, piece_rect)


def main():
    clock = pygame.time.Clock()
    game = GameManager()
    state = "menu"  # can be "menu", "game", "options", "quit"
    selected_square = None

    
    PLAY_BUTTON = Button(image=pygame.Surface((150, 50)), pos=(320, 250), 
                        text_input="PLAY", font=pygame.font.SysFont('MS Sans Serif Regular', 30), base_color="#d7fcd4", hovering_color="White")
    OPTIONS_BUTTON = Button(image=pygame.Surface((150, 50)), pos=(320, 400), 
                        text_input="OPTIONS", font=pygame.font.SysFont('MS Sans Serif Regular', 30), base_color="#d7fcd4", hovering_color="White")
    QUIT_BUTTON = Button(image=pygame.Surface((150, 50)), pos=(320, 550), 
                        text_input="QUIT", font=pygame.font.SysFont('MS Sans Serif Regular', 30), base_color="#d7fcd4", hovering_color="White")
    MENU_BUTTON = Button(image=pygame.Surface((150, 50)), pos=(320, 500), 
                        text_input="GO TO MAIN MENU", font=pygame.font.SysFont('MS Sans Serif Regular', 30), base_color="#d7fcd4", hovering_color="White")

    while state != "quit":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state = "quit"

            # --- Menu state ---
            if state == "menu":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if PLAY_BUTTON.checkForInput(mouse_pos):
                        state = "game"
                    elif OPTIONS_BUTTON.checkForInput(mouse_pos):
                        state = "options"
                    elif QUIT_BUTTON.checkForInput(mouse_pos):
                        state = "quit"
                        
            # --- Game state ---
            elif state == "game":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    row, col = y // TILE_SIZE, x // TILE_SIZE
                    if selected_square is None:
                        piece = game.board.getPieceAt((row, col))
                        if piece and piece.colour == game.board.turn:
                            selected_square = (row, col)
                    else:
                        start_pos = selected_square
                        end_pos = (row, col)
                        game.board.movePiece(start_pos, end_pos, switch_turn=True)
                        selected_square = None
                
                if game.isCheckMate():
                    state = "checkmate"
                if game.isStaleMate():
                    state = "stalemate"

            if state == "checkmate" or state == "stalemate":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if MENU_BUTTON.checkForInput(mouse_pos):
                        state = "menu"
                        game.restartBoard()

        if state == "checkmate":
            # Semi-transparent overlay
            overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
            overlay.fill("black")
            overlay.set_alpha(180)  # transparency
            SCREEN.blit(overlay, (0, 0))

            # Fonts
            title_font = pygame.font.SysFont("MS Sans Serif Regular", 50, bold=True)
            subtitle_font = pygame.font.SysFont("MS Sans Serif Regular", 30)

            # Texts
            title_text = title_font.render("Checkmate!", True, (255, 215, 0))  # gold color
            subtitle_text = subtitle_font.render(f"{game.board.turn} lost.", True, (255, 255, 255))

            # Rects (centered)
            title_rect = title_text.get_rect(center=(BOARD_SIZE / 2, 240))
            subtitle_rect = subtitle_text.get_rect(center=(BOARD_SIZE / 2, 280))

            # Background box behind texts & button
            box_width, box_height = 400, 250
            box_rect = pygame.Rect(0, 0, box_width, box_height)
            box_rect.center = (BOARD_SIZE / 2, BOARD_SIZE / 2)
            pygame.draw.rect(SCREEN, (40, 40, 40), box_rect, border_radius=20)
            pygame.draw.rect(SCREEN, (200, 200, 200), box_rect, width=3, border_radius=20)

            # Draw texts
            SCREEN.blit(title_text, title_rect)
            SCREEN.blit(subtitle_text, subtitle_rect)

            # Menu button centered below text
            MENU_BUTTON.rect.center = (BOARD_SIZE / 2, 340)
            MENU_BUTTON.changeColor(mouse_pos)
            MENU_BUTTON.update(SCREEN)

            pygame.display.update()

            

        elif state == "menu":
            SCREEN.fill('black')

            mouse_pos = pygame.mouse.get_pos()

            MENU_TEXT = pygame.font.SysFont('MS Sans Serif Regular', 30).render("MAIN MENU", True, "#b68f40")
            MENU_RECT = MENU_TEXT.get_rect(center=(BOARD_SIZE / 2, 100))

            SCREEN.blit(MENU_TEXT, MENU_RECT)

            for button in [PLAY_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
                button.changeColor(mouse_pos)
                button.update(SCREEN)

            pygame.display.update()

        elif state == "game":
            draw_board(game.board, selected_square)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


main()