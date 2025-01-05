async function selectGame(gameName) {
    try {
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ game: gameName }),
        });

        const data = await response.json();
        console.log(data.game)

        window.location.href = `/${gameName}`;
    } catch (error) {
        console.error('Error selecting game:', error);
    }
}

document.getElementById('tictactoe-btn').addEventListener('click', () => selectGame('tictactoe'));
document.getElementById('connect4-btn').addEventListener('click', () => selectGame('connect4'));
