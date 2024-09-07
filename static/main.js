document.addEventListener('DOMContentLoaded', async () => {
    await getPrediction();  // Call the getPrediction function when the page loads
});

async function getPrediction() {
    try {
        const response = await fetch('/predict', {
            method: 'POST'
        });
        const data = await response.json();
        document.getElementById('output').innerText = `Prediction: ${data.prediction}`;
        
        // Additional JS logic to work with the prediction data
        console.log('Received Prediction Data:', data.prediction);
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }       
}
