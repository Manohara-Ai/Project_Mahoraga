from flask import Flask, render_template, jsonify, request, url_for
import numpy as np
import random
import torch
import os

from alpha_zero import ResNet, MCTS

app = Flask("__name__")

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        self.state = self.get_initial_state()
        self.game_over = False

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        self.player = random.choice([1, -1])
        self.turn = 'player' if self.player == 1 else 'bot'
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = TicTacToe()
model = ResNet(game, 4, 64, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 2,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

model_path = f"./models/model_{game}.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
mcts = MCTS(game, args, model)

@app.route('/')
def home():
    game.__init__()
    game_constants = {
        'player': game.player,
        'bot': -game.player,
    }
    if request.args.get('format') == 'json':
        return jsonify(game_constants)
    return render_template('index.html')

@app.route('/send_game_state')
def send_game_state():
    return jsonify(
        {
            'state': (game.state).tolist(), 
            'game_over': game.game_over,
            'turn': game.turn,
        }
    )

@app.route('/get_player_move', methods=['POST'])
def player_move():
    if game.game_over == True:
        return jsonify({"message": "Game Over! Refresh to continue playing!!", "game_over": True}), 400
    if game.turn == 'player' and game.game_over == False:
        data = request.get_json()
        index = data['cellIndex']
        row, col = index//3, index % 3
        game.state[row][col] = game.player
        value, game.game_over = game.get_value_and_terminated(game.state, index)
        if game.game_over:
            if value  == 1:
                return jsonify({"message": "Player Wins! Refresh to continue playing!!", "game_over": True}), 400
            else:
                return jsonify({"message": "Tie! Refresh to continue playing!!", "game_over": True}), 400
        game.turn = 'bot'
        return jsonify({"message": "Move registered successfully!", "game_over": False}), 200
    else:
        return jsonify({"message": "Not the player's turn!"}), 400

@app.route('/push_ai_move')
def ai_move():
    if game.game_over == True:
        return jsonify({"message": "Game Over! Refresh to continue playing!!", "game_over": True}), 400
    if game.turn == 'bot' and game.game_over == False:
        neutral_state = game.change_perspective(game.state, game.get_opponent(game.player))
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        game.state = game.get_next_state(game.state, action, game.get_opponent(game.player))
        value, game.game_over = game.get_value_and_terminated(game.state, action)
        if game.game_over:
            if value  == 1:
                return jsonify({"message": "Ai Wins! Refresh to continue playing!!", "game_over": True}), 400
            else:
                return jsonify({"message": "Tie! Refresh to continue playing!!", "game_over": True}), 400
        game.turn = 'player'
        return jsonify({"message": "AI made its move!", "game_over": False}), 200
    else:
        return jsonify({"message": "Not the bot's turn!"}), 400


if __name__ == '__main__':
    app.run()