import pygame
import py_button
from pyPath import path

# Variabel buat display window
screenWidth = 800
screenHeight = 500

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption('Interface from python')

# Comic Image
comic1 = pygame.image.load(path + 'Comic1.jpg')
comic2 = pygame.image.load(path + 'Comic2.jpg')
comic3 = pygame.image.load(path + 'Comic3.jpg')
comic4 = pygame.image.load(path + 'Comic4.jpg')
comic5 = pygame.image.load(path + 'Comic5.jpg')
comic1 = pygame.transform.scale(comic1, (400, 300))
comic2 = pygame.transform.scale(comic2, (400, 300))
comic3 = pygame.transform.scale(comic3, (400, 300))
comic4 = pygame.transform.scale(comic4, (400, 300))
comic5 = pygame.transform.scale(comic5, (400, 300))
index = 0
comicArray = [comic1, comic2, comic3, comic4, comic5]

# Button - with image prop w="362" h="126"
backButtonImage = pygame.image.load(path + 'py_back.png').convert_alpha()
nextButtonImage = pygame.image.load(path + 'py_next.png').convert_alpha()

backButton = py_button.Button(
    100,
    450,
    backButtonImage,
    100/362
)

nextButton = py_button.Button(
    (screenWidth - 100) - (nextButtonImage.get_width() * 100/362),
    450,
    nextButtonImage,
    100/362
)

# Game loop
run = True
while run:
    screen.fill((0, 0, 0))

  # Button callback
    if backButton.draw(screen):
        if index != 0:
          index = index-1
        print('Trigger Back')
        print('sekarang halaman', index + 1)

    if nextButton.draw(screen):
        if index != 4:
          index = index+1
        print('Trigger Next')
        print('sekarang halaman', index + 1)

    screen.blit(
      comicArray[index],
      ((screenWidth/2) - 200, (50))
    )

    # Window event handler
    for event in pygame.event.get():
        # Quit button
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()
