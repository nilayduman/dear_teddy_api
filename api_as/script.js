async function analyzeText() {
    const inputText = document.getElementById("inputText").value;
    const resultDiv = document.getElementById("result");
    const sentiment = document.getElementById("sentiment");
    const emotion = document.getElementById("emotion");
    const animationDiv = document.getElementById("animation");

    if (!inputText.trim()) {
        alert("Lütfen bir metin girin.");
        return;
    }

    try {
        // API isteği yap
        const response = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: inputText })
        });

        const data = await response.json();

        // Sonuçları işleme
        sentiment.innerHTML = `Duygu: ${data.sentiment}`;
        emotion.innerHTML = `Duygusal Durum: ${JSON.stringify(data.emotions)}`;


        resultDiv.style.display = 'block';
    } catch (error) {
        console.error("Hata:", error);
    }
}

