let player;
let bot;
let turn;
let state;
let message;
let game_over;
let playerMoveEnabled = false;  // A flag to prevent multiple calls to enable player move

async function initializeGame() {
    try {
        // Fetch game constants (player and bot)
        const response = await fetch('/?format=json');
        const data = await response.json();

        player = data.player;
        bot = data.bot;

        console.log("Player:", player);
        console.log("Bot:", bot);

        await fetchGameState();
    } catch (error) {
        console.error('Error fetching game constants:', error);
    }
}

async function fetchGameState() {
    try {
        const response = await fetch('/send_game_state');
        const data = await response.json();
        
        turn = data.turn;
        state = data.state; 
        game_over = data.game_over // Set the turn to the fetched value
        console.log("Current turn:", turn);
        console.log("Current state:", state);
        updateBoard();
        if (game_over == false){
            if (turn === 'player') {
                if (!playerMoveEnabled) {
                    enablePlayerMoves();
                    playerMoveEnabled = true;
                }
            } else {
                handleAIMove();
            }
        }
    } catch (error) {
        console.error('Error fetching game state:', error);
        document.getElementById('board').textContent = 'Failed to load game state.';
    }
}

function enablePlayerMoves() {
    const cells = document.querySelectorAll(".cell");

    cells.forEach((cell, index) => {
        cell.addEventListener("click", handleCellClick);
    });
}

function updateBoard() {
    const cells = document.querySelectorAll(".cell");

    state.forEach((row, rowIndex) => {
        row.forEach((value, colIndex) => {
            const index = rowIndex * 3 + colIndex;  // Convert 2D indices to 1D index
            const cell = cells[index];
            
            if (value === 1) {
                cell.textContent = 'X';  // Mark with 'X' for player
                cell.classList.add('x-player');
                cell.classList.remove('o-player');
            } else if (value === -1) {
                cell.textContent = 'O';  // Mark with 'O' for bot
                cell.classList.add('o-player');
                cell.classList.remove('x-player');
            } else {
                cell.textContent = '';  // Empty cell
                cell.classList.remove('x-player', 'o-player');
            }
        });
    });
}

function handleCellClick(event) {
    const cell = event.target;
    const index = Array.from(cell.parentElement.children).indexOf(cell);

    if (cell.textContent === '') {
        makePlayerMove(index);  // Pass the index of the clicked cell
    }
}

async function makePlayerMove(index) {
    try {
        const response = await fetch('/get_player_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cellIndex: index
            })
        });
        const data = await response.json();
        message = data.message;
        game_over = data.game_over
        console.log("Game Over: ", game_over)
        console.log("Message: ", message)
        if (game_over==true){
            renderGameMessage(message, game_over)
        }
        await fetchGameState();
    } catch (error) {
        console.error('Error making player move:', error);
    }
}

async function handleAIMove() {
    try {
        const response = await fetch('/push_ai_move');
        const data = await response.json();
        message = data.message;
        game_over = data.game_over
        console.log("Game Over: ", game_over)
        console.log("Message: ", message)
        if (game_over==true){
            renderGameMessage(message, game_over)
        }
        await fetchGameState();
    } catch (error) {
        console.error('Error fetching AI moves:', error);
        document.getElementById('board').textContent = 'Failed to get AI moves.';
    }
}

function renderGameMessage(message, gameOver) {
    const resultElement = document.querySelector('.game-result');
    
    // Sanitize the message to use as a valid class name
    const sanitizedMessage = message.toLowerCase().replace(/\s+/g, '-');  // Replace spaces with hyphens
    
    resultElement.textContent = message;
    resultElement.classList.add(sanitizedMessage);  // Add the sanitized message as a class

    // Disable both player and AI moves when the game is over
    if (gameOver) {
        disablePlayerMoves();
    }
}

function disablePlayerMoves() {
    const cells = document.querySelectorAll(".cell");
    cells.forEach(cell => {
        cell.removeEventListener("click", handleCellClick);  // Remove the click listener
        cell.style.pointerEvents = "none";  // Disable the click event
    });
}

// Initialize the game
initializeGame();
