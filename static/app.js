const form = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const messages = document.querySelector('#messages');
const activity = document.querySelector('#activity');
const welcome = document.querySelector('#welcome');
const history = [];

function addActivity(label, detail, icon = '↗') {
  const empty = activity.querySelector('.activity-empty');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = 'activity-item';
  item.innerHTML = `<span class="icon">${icon}</span><div>${label}<small>${detail}</small></div>`;
  activity.append(item);
}

function addMessage(role, content = '') {
  welcome?.remove();
  const element = document.createElement('div');
  element.className = `message ${role}`;
  element.innerHTML = `<div class="message-label">${role === 'user' ? 'You' : 'Local agent'}</div><p></p>`;
  element.querySelector('p').textContent = content;
  messages.append(element);
  messages.scrollTop = messages.scrollHeight;
  return element.querySelector('p');
}

async function sendMessage(value) {
  const text = value.trim();
  if (!text) return;
  addMessage('user', text);
  history.push({ role: 'user', content: text });
  prompt.value = '';
  prompt.style.height = 'auto';
  addActivity('Thinking', 'Planning the next step', '◌');
  const assistant = addMessage('assistant');
  let answer = '';
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text, history: history.slice(0, -1) }) });
    if (!response.ok || !response.body) throw new Error('Could not connect to the local agent.');
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    while (true) {
      const { value: chunk, done } = await reader.read();
      if (done) break;
      buffer += chunk;
      const events = buffer.split('\n\n');
      buffer = events.pop();
      for (const raw of events) {
        if (!raw.startsWith('data: ')) continue;
        const data = JSON.parse(raw.slice(6));
        if (data.type === 'status') addActivity(data.label, data.detail, '◌');
        if (data.type === 'tool') addActivity(data.label, data.detail, '↗');
        if (data.type === 'message') { answer = data.content; assistant.textContent = answer; messages.scrollTop = messages.scrollHeight; }
        if (data.type === 'error') { answer = data.message; assistant.textContent = answer; addActivity('Needs setup', 'See the terminal for Ollama setup', '!'); }
      }
    }
    history.push({ role: 'assistant', content: answer });
  } catch (error) { assistant.textContent = error.message; addActivity('Connection error', 'The server could not be reached', '!'); }
}

form.addEventListener('submit', (event) => { event.preventDefault(); sendMessage(prompt.value); });
prompt.addEventListener('input', () => { prompt.style.height = 'auto'; prompt.style.height = `${Math.min(prompt.scrollHeight, 130)}px`; });
prompt.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => sendMessage(button.dataset.prompt)));
document.querySelector('#new-chat').addEventListener('click', () => window.location.reload());