# Project_Mahoraga

**Project_Mahoraga** is an AI-powered game enhancement project that leverages advanced artificial intelligence techniques to improve the performance and decision-making of games. The project’s core is built around **AlphaZero**, a self-learning AI system that uses **reinforcement learning** to master game strategies, paired with **Monte Carlo Tree Search (MCTS)** for optimizing decision-making in real-time. This AI-driven system is designed to continuously improve its gameplay performance as it learns from interactions.

## Table of Contents

1. [Project Overview](#project-overview)
   - [Frontend](#frontend)
   - [Backend](#backend)
2. [Tic-Tac-Toe Game](#tic-tac-toe-game)
3. [AlphaZero and MCTS](#alphazero-and-mcts)
   - [AlphaZero](#alphazero)
   - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
4. [Future Plans](#future-plans)
   - [Adding More Games](#adding-more-games)
   - [Model Improvement](#model-improvement)
   - [Adaptive AI](#adaptive-ai)

## Project Overview

The project focuses on creating a platform where users can play games against an AI agent powered by AlphaZero and MCTS. The AI improves over time by learning from its own experience and refining its strategies through reinforcement learning. Initially, the project features a **Tic-Tac-Toe** game where users can play against the AI, but the framework is designed to be easily extended to other games.

### Frontend

The frontend is built using **HTML**, **CSS**, and **JavaScript**. It provides an interactive interface for users to play Tic-Tac-Toe against the AI. The frontend communicates with the backend using HTTP routes to send and receive game data, allowing real-time interaction between the user and the AI.

### Backend

The backend is powered by **Python**, where the game logic and AI training are handled. The AI model is implemented using AlphaZero and Monte Carlo Tree Search (MCTS). Python and JavaScript communicate through API routes, allowing smooth interaction between the user’s actions on the frontend and the AI’s decision-making process on the backend.

## Tic-Tac-Toe Game

At the core of the project is a **Tic-Tac-Toe** game, where users can play against an AI that learns and improves as it plays. The game is simple but serves as a demonstration of the power of AlphaZero and MCTS. The AI adapts its strategy based on its experience, becoming more skilled at predicting the best possible moves over time.

The game board is rendered on the frontend, and users can make moves by clicking on the cells. The backend AI evaluates the best move using MCTS and updates the board accordingly. The AI continuously learns, becoming more efficient in its decision-making as it plays more games.

## AlphaZero and MCTS

### AlphaZero

The AI in Project_Mahoraga is inspired by **AlphaZero**, a deep reinforcement learning algorithm developed by DeepMind. AlphaZero learns to play games by playing against itself, improving its strategies through trial and error. Unlike traditional game AI, which relies on pre-programmed strategies, AlphaZero learns from scratch and develops its own optimal strategies.

In Mahoraga, AlphaZero is used to power the game-playing agent. It learns and refines its Tic-Tac-Toe strategy through self-play, improving with every game it plays.

### Monte Carlo Tree Search (MCTS)

To further enhance decision-making, Mahoraga utilizes **Monte Carlo Tree Search (MCTS)**. MCTS is an AI algorithm that simulates multiple future game states, evaluating the potential outcomes of each possible move to select the best one. This allows the AI to assess a wide range of scenarios and make more informed decisions.

Together, AlphaZero and MCTS create a powerful system where the AI learns from its own experiences and evaluates potential moves with high efficiency.

## Future Plans

The long-term goal of Project_Mahoraga is to expand the range of games that the AI can play and improve its learning process. Some future plans for the project include:

### Adding More Games

In addition to Tic-Tac-Toe, we plan to implement other classic board games like **Connect Four**, **Chess**, and potentially even more complex strategy games. The goal is to create an AI that can handle a variety of game environments and strategies.

### Model Improvement

One of the most exciting features we plan to add is the ability for the model to **train in real-time**. As people play games against the AI, the model will continue to learn and refine its strategies based on actual gameplay, improving with every interaction.

### Adaptive AI

The AI will not only learn from self-play but also from interactions with human players, allowing it to adapt to different playing styles and become even more challenging over time.

With these improvements, Project_Mahoraga will evolve into a robust platform where the AI continuously grows and refines its strategies across multiple games.
