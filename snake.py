import pygame
import numpy as np
import random

# Define constants
GRID_SIZE = 15
CELL_SIZE = 15
GRID_WIDTH = 15
GRID_HEIGHT = 15
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FPS = 50

# Initialize pygame
pygame.init()
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game Q-learning")

# Define Snake class
class Snake:
    def __init__(self):
        self.snake_body = [(5, 5)]
        self.direction = (1, 0)
        self.apple_position = self.generate_apple()
        self.score = 0
        self.reward=0


    def generate_apple(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in self.snake_body:
                return x, y

    def move(self):
        current_head = self.snake_body[0]
        new_head = ((current_head[0] + self.direction[0]) % GRID_WIDTH, (current_head[1] + self.direction[1]) % GRID_HEIGHT)
        if new_head in self.snake_body[1:]:
            return True  # Snake bites itself
        self.snake_body.insert(0, new_head)
        if new_head == self.apple_position:
            self.apple_position = self.generate_apple()
            self.score += 1
        else:
            self.snake_body.pop()
        return False

    def reset(self):
        self.snake_body = [(10, 10)]
        self.direction = (1, 0)
        self.apple_position = self.generate_apple()
        self.score = 0

    def turn(self, direction):
        if direction == 'UP' and self.direction != (0, 1):
            self.direction = (0, -1)
        elif direction == 'DOWN' and self.direction != (0, -1):
            self.direction = (0, 1)
        elif direction == 'LEFT' and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif direction == 'RIGHT' and self.direction != (-1, 0):
            self.direction = (1, 0)

    def get_state(self):
        # Represent the state as a binary array:
        # [obstacle_up, obstacle_down, obstacle_left, obstacle_right, apple_up, apple_down, apple_left, apple_right]
        head_x, head_y = self.snake_body[0]
        apple_x, apple_y = self.apple_position
        state = [head_y == 0, head_y == GRID_HEIGHT - 1, head_x == 0, head_x == GRID_WIDTH - 1,
                 head_y > apple_y, head_y < apple_y, head_x > apple_x, head_x < apple_x]
        return np.array(state, dtype=int)

    def get_reward(self):
        if self.move():
            return -1  # Penalize if the snake dies or makes a move
        elif self.snake_body[0] == self.apple_position:
            self.score += 10  # Increment score
            return 1  # Reward if the snake eats an apple
        else:
            head_x, head_y = self.snake_body[0]
            apple_x, apple_y = self.apple_position
            current_distance = abs(head_x - apple_x) + abs(head_y - apple_y)

            # Move closer to the apple
            self.move()
            head_x, head_y = self.snake_body[0]
            new_distance = abs(head_x - apple_x) + abs(head_y - apple_y)

            # Calculate reward based on the change in distance
            self.reward = 0
            if new_distance < current_distance:
                self.reward = 0.1  # Encourage moving closer to the apple
            elif new_distance > current_distance:
                self.reward = -0.1  # Discourage moving away from the apple
            return self.reward

# Define Q-learning agent
class QLearningAgent:
    def __init__(self, epsilon_decay=0.999, lr_decay=0.001):
        self.q_table = np.zeros((256, 4))  # Q-table with 256 states and 4 actions
        self.learning_rate = 0.001
        self.discount_factor = 0.01
        self.epsilon = 0.9
        self.epsilon_decay = epsilon_decay
        self.lr_decay = lr_decay

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state][action] += self.learning_rate * \
            (reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state][action])
        self.epsilon *= self.epsilon_decay
        self.learning_rate *= self.lr_decay

# Initialize game and agent
snake_game = Snake()
agent = QLearningAgent()

# Training loop
for epoch in range(100000):  # Train for 1000 epochs
    game_over = False
    snake_game.reset()
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        state = snake_game.get_state()
        action = agent.get_action(state)
        snake_game.turn(['UP', 'DOWN', 'LEFT', 'RIGHT'][action])
        reward = snake_game.get_reward()
        next_state = snake_game.get_state()
        agent.update_q_table(state, action, reward, next_state)

        print(f"next_state: {next_state}, Score: {snake_game.score}, Reward: {reward}")

        if reward == -1:
            game_over = True

        # Draw the game during training
        screen.fill(WHITE)
        for segment in snake_game.snake_body:
            pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, (snake_game.apple_position[0] * CELL_SIZE, snake_game.apple_position[1] * CELL_SIZE,
                                        CELL_SIZE, CELL_SIZE))
        score_text = font.render(f"Score: {snake_game.score}", True, BLACK)
        screen.blit(score_text, (5, 5))
        pygame.display.update()
        clock.tick(FPS)

    # Print epoch information
    print(f"Epoch: {epoch}, Score: {snake_game.score}, Reward: {reward}")

# Testing the trained model
FPS = 5
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    state = snake_game.get_state()
    action = agent.get_action(state)
    snake_game.turn(['UP', 'DOWN', 'LEFT', 'RIGHT'][action])
    reward = snake_game.get_reward()

    if reward != -1:
        reward = -0.01

    if reward == -1:
        print("Game over!")
        break

    # Draw the game during testing
    screen.fill(WHITE)
    for segment in snake_game.snake_body:
        pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (snake_game.apple_position[0] * CELL_SIZE, snake_game.apple_position[1] * CELL_SIZE,
                                    CELL_SIZE, CELL_SIZE))
    score_text = font.render(f"Score: {snake_game.score}", True, BLACK)
    screen.blit(score_text, (5, 5))
    pygame.display.update()
    clock.tick(FPS)
