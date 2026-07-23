// Novel Assistant Logic
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatWindow = document.getElementById('chat-window');

function appendMessage(text, sender, sources=null) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    
    // Avatar
    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('avatar');
    avatarDiv.textContent = sender === 'user' ? 'U' : 'AI';
    msgDiv.appendChild(avatarDiv);

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = text; // Allow basic HTML (like bold)
    
    if (sources && sources.length > 0) {
        const sourceDiv = document.createElement('div');
        sourceDiv.classList.add('sources');
        let html = '<strong>Sources:</strong><br>';
        sources.forEach(s => {
            html += `<em>${s.title} - ${s.location}</em><br>`;
        });
        sourceDiv.innerHTML = html;
        contentDiv.appendChild(sourceDiv);
    }
    
    msgDiv.appendChild(contentDiv);
    
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    return msgDiv; // Return so we can remove loading state
}

async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    
    appendMessage(msg, 'user');
    chatInput.value = '';
    
    const loadingDiv = appendMessage('Thinking...', 'assistant');
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: msg })
        });
        const data = await response.json();
        
        chatWindow.removeChild(loadingDiv);
        
        if (data.error) {
            appendMessage("Error: " + data.error, 'assistant');
        } else {
            appendMessage(data.answer, 'assistant', data.sources);
        }
    } catch (err) {
        chatWindow.removeChild(loadingDiv);
        appendMessage("Failed to connect to server.", 'assistant');
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Check API status
fetch('/api/status').then(res => res.json()).then(data => {
    if (!data.novel) {
        document.getElementById('status-indicator').style.color = '#ef4444';
        document.getElementById('status-indicator').nextSibling.textContent = ' Offline (Needs Ingestion)';
    }
}).catch(err => {
    document.getElementById('status-indicator').style.color = '#ef4444';
    document.getElementById('status-indicator').nextSibling.textContent = ' Server Offline';
});
