import os
import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

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
FPS = 200

# Initialize pygame
pygame.init()
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game Deep Q-Learning")

# Define Snake class
class Snake:
    def __init__(self):
        self.snake_body = [(10, 10)]
        self.direction = (1, 0)
        self.apple_position = self.generate_apple()
        self.score = 0
        self.reward = 0

    def generate_apple(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in self.snake_body:
                return x, y

    def move(self):
        current_head = self.snake_body[0]
        new_head = (
            (current_head[0] + self.direction[0]) % GRID_WIDTH, (current_head[1] + self.direction[1]) % GRID_HEIGHT)
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
        directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        new_direction = directions.get(direction)
        if new_direction and new_direction != (-self.direction[0], -self.direction[1]):
            self.direction = new_direction

    def get_state(self):
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


# Define Deep Q-Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)  # Added layer
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define named tuple for transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Initialize game and DQN
snake_game = Snake()
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Define epsilon-greedy exploration
EPS_START = 0.9
EPS_END = 0.5
EPS_DECAY = 200
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax().view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], dtype=torch.long)


# Training loop
BATCH_SIZE = 68
GAMMA = 0.999
TARGET_UPDATE = 10
memory = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
    optimizer.step()


# Check if pre-trained model exists
pretrained_model_path = 'best_snake_model.pth'
if os.path.exists(pretrained_model_path):
    print("Loading pre-trained model...")
    policy_net.load_state_dict(torch.load(pretrained_model_path))
else:
    print("No pre-trained model found. Starting training from scratch...")

for epoch in range(1000):  # Train for 1000 epochs
    game_over = False
    snake_game.reset()
    state = torch.tensor([snake_game.get_state()], dtype=torch.float32)
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        action = select_action(state)
        snake_game.turn(['UP', 'DOWN', 'LEFT', 'RIGHT'][action.item()])
        reward = snake_game.get_reward()
        score = snake_game.score
        new_state = torch.tensor([snake_game.get_state()], dtype=torch.float32)
        if reward == -1:
            game_over = True
            new_state = None
        memory.append(Transition(state, action, new_state, torch.tensor([reward], dtype=torch.float32)))
        state = new_state

        optimize_model()

        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Draw the game during training
        screen.fill(WHITE)
        for segment in snake_game.snake_body:
            pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED,
                         (snake_game.apple_position[0] * CELL_SIZE, snake_game.apple_position[1] * CELL_SIZE,
                          CELL_SIZE, CELL_SIZE))
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))
        pygame.display.update()
        clock.tick(FPS)

    # Print epoch information
    print(f"Epoch: {epoch}, Score: {score}, Reward: {reward}")

# Testing the trained model
FPS = 5
best_score = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    state = torch.tensor([snake_game.get_state()], dtype=torch.float32)
    action = policy_net(state).argmax().item()
    snake_game.turn(['UP', 'DOWN', 'LEFT', 'RIGHT'][action])
    reward = snake_game.get_reward()
    score = snake_game.score

    if reward != -1:
        reward = -0.01

    if reward == -1:
        print(f"Game over! Score: {score}")
        break

    # Draw the game during testing
    screen.fill(WHITE)
    for segment in snake_game.snake_body:
        pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (snake_game.apple_position[0] * CELL_SIZE, snake_game.apple_position[1] * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE))
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))
    pygame.display.update()
    clock.tick(FPS)

    # Save the best model
    if score > best_score:
        best_score = score
        torch.save(policy_net.state_dict(), 'best_snake_model.pth')
        print("Model saved !")
