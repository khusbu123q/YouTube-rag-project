// ⚠️ YOUR NGROK URL
const BACKEND_URL = "https://detection-wife-cattail.ngrok-free.dev";
let currentVideoId = null;

chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const url = tabs[0].url;

  if (!url || !url.includes("youtube.com/watch")) {
    setStatus("❌ Please open a YouTube video first!");
    return;
  }

  currentVideoId = new URL(url).searchParams.get("v");
  setStatus(`⏳ Indexing video: ${currentVideoId}...`);

  fetch(`${BACKEND_URL}/health`, {
    headers: { "ngrok-skip-browser-warning": "true" }
  })
  .then(res => res.json())
  .then(() => {
    return fetch(`${BACKEND_URL}/index`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
      },
      body: JSON.stringify({ video_id: currentVideoId })
    });
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "success" || data.status === "already_indexed") {
      setStatus("✅ Ready! Ask anything about this video.");
      enableInput();
    } else {
      setStatus(`❌ ${data.message}`);
    }
  })
  .catch(() => {
    setStatus("❌ Backend not running! Check Colab.");
  });
});

document.getElementById("ask-btn").addEventListener("click", askQuestion);
document.getElementById("question").addEventListener("keypress", (e) => {
  if (e.key === "Enter") askQuestion();
});

function askQuestion() {
  const questionInput = document.getElementById("question");
  const question = questionInput.value.trim();
  if (!question || !currentVideoId) return;

  const chatBox = document.getElementById("chat-box");
  const welcome = chatBox.querySelector(".welcome");
  if (welcome) welcome.remove();

  chatBox.innerHTML += `<div class="user-msg">🧑 ${question}</div>`;
  chatBox.innerHTML += `<div class="loading" id="loading">🤖 Thinking...</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;

  questionInput.value = "";
  disableInput();

  fetch(`${BACKEND_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true"
    },
    body: JSON.stringify({ video_id: currentVideoId, question: question })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("loading").remove();
    chatBox.innerHTML += `<div class="bot-msg">🤖 ${data.answer}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
    enableInput();
  })
  .catch(() => {
    document.getElementById("loading").remove();
    chatBox.innerHTML += `<div class="bot-msg">❌ Error connecting to backend.</div>`;
    enableInput();
  });
}

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

function enableInput() {
  document.getElementById("question").disabled = false;
  document.getElementById("ask-btn").disabled = false;
  document.getElementById("question").focus();
}

function disableInput() {
  document.getElementById("question").disabled = true;
  document.getElementById("ask-btn").disabled = true;
}