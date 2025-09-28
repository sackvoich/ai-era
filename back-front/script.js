const API_CONFIG = {
    baseURL: 'http://127.0.0.1:8000',
    endpoints: {
        predict: '/predict',
        health: '/health'
    },
};

// --- Элементы DOM ---
const textInput = document.getElementById('textInput');
const checkBtn = document.getElementById('checkBtn');
const resultDiv = document.getElementById('result');
const modelInfoDiv = document.querySelector('.model-info');

// Элементы для переключения режимов
const textModeBtn = document.getElementById('textModeBtn');
const fileModeBtn = document.getElementById('fileModeBtn');
const textInputPanel = document.getElementById('textInputPanel');
const fileInputPanel = document.getElementById('fileInputPanel');

// Элементы для загрузки файла
const fileInput = document.getElementById('fileInput');
const fileSelectBtn = document.getElementById('fileSelectBtn');
const fileStatus = document.getElementById('fileStatus');
const fileUploadArea = document.querySelector('.file-upload-area');


// --- Логика переключения режимов ---
function switchMode(mode) {
    if (mode === 'text') {
        textModeBtn.classList.add('active');
        fileModeBtn.classList.remove('active');
        textInputPanel.style.display = 'block';
        fileInputPanel.style.display = 'none';
    } else if (mode === 'file') {
        fileModeBtn.classList.add('active');
        textModeBtn.classList.remove('active');
        fileInputPanel.style.display = 'block';
        textInputPanel.style.display = 'none';
    }
}

textModeBtn.addEventListener('click', () => switchMode('text'));
fileModeBtn.addEventListener('click', () => switchMode('file'));


// --- Логика API ---
async function checkTextForProfanity(text) {
    try {
        const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.predict}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        return {
            hasProfanity: data.label === 1,
            probability: Math.round(data.score * 100),
        };
    } catch (error) {
        console.error('Ошибка API:', error);
        return { isError: true, message: 'Не удалось связаться с моделью' };
    }
}

// --- Отображение результата ---
function displayResult(result) {
    if (result.isError) {
        resultDiv.innerHTML = `<div class="result-content" style="color: #dc3545;">${result.message}</div>`;
        return;
    }

    const status = result.hasProfanity ? '🚫 ОБНАРУЖЕН МАТ (1)' : '✅ ЧИСТЫЙ ТЕКСТ (0)';
    const statusClass = result.hasProfanity ? 'status-profanity' : 'status-clean';
    
    resultDiv.innerHTML = `
        <div class="result-content">
            <div class="result-status ${statusClass}">${status}</div>
            <div class="result-probability">Вероятность: ${result.probability}%</div>
        </div>
    `;
}

// --- Основная функция проверки ---
async function handleCheck() {
    const text = textInput.value.trim();
    if (!text) {
        alert('Пожалуйста, введите текст для проверки.');
        return;
    }
    if (text.length > 4000) {
        alert('Текст слишком длинный (максимум 4000 символов).');
        return;
    }
    
    resultDiv.innerHTML = '<div class="result-content">🔍 Модель анализирует текст...</div>';
    checkBtn.disabled = true;
    
    try {
        const result = await checkTextForProfanity(text);
        displayResult(result);
    } finally {
        checkBtn.disabled = false;
    }
}

checkBtn.addEventListener('click', handleCheck);
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        handleCheck();
    }
});


// --- Логика работы с файлами ---
fileSelectBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

async function handleFile(file) {
    if (!file || !file.type.startsWith('text/')) {
        fileStatus.textContent = 'Пожалуйста, выберите текстовый файл (.txt)';
        return;
    }

    fileStatus.textContent = `Чтение файла: ${file.name}...`;
    try {
        const text = await readFile(file);
        textInput.value = text;
        fileStatus.innerHTML = `<span class="file-success">${file.name} ✅ загружен</span>`;
        
        // Переключаемся на вкладку с текстом и автоматически запускаем проверку
        switchMode('text');
        setTimeout(handleCheck, 300);

    } catch (error) {
        fileStatus.innerHTML = `<span class="file-error">Ошибка чтения файла</span>`;
        console.error('Ошибка чтения файла:', error);
    }
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file, 'UTF-8');
    });
}

// Drag & Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    fileUploadArea.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    fileUploadArea.addEventListener(eventName, () => fileUploadArea.classList.add('drag-over'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    fileUploadArea.addEventListener(eventName, () => fileUploadArea.classList.remove('drag-over'), false);
});

fileUploadArea.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
}, false);


// --- Инициализация и проверка статуса API ---
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.health}`);
        if (!response.ok) throw new Error('API not responding');
        const data = await response.json();
        console.log('API status:', data);
    } catch (error) {
        console.error('API недоступен:', error);
        resultDiv.innerHTML = '<div class="result-content" style="color: #dc3545;">⚠️ Ошибка: не удалось подключиться к серверу с моделью.</div>';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    switchMode('text'); // Устанавливаем начальный режим
    checkAPIStatus();
});