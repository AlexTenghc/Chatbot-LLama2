const form = document.getElementById('user-input-form');
const userInput = document.getElementById('user-input');
const messages = document.getElementById('messages');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const userMessage = userInput.value.trim();
    if (userMessage === '') return;

    appendMessage(userMessage, 'sent');
    userInput.value = '';
    scrollToBottom();

    fetch('/get_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `user_input=${encodeURIComponent(userMessage)}`,
    })
        .then((response) => response.text())
        .then((data) => {
            appendMessage(data, 'received');
            scrollToBottom();
        });
});

function appendMessage(message, type) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', type);
    messageElement.textContent = message;
    messages.appendChild(messageElement);
}

function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
}
