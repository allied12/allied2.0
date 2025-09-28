// dashboard/01.js
document.addEventListener('DOMContentLoaded', () => {

    // --- Script for Dynamic Date and Toggle Switch ---
    const dateElement = document.getElementById('challan-date');
    if (dateElement) {
        dateElement.textContent = new Date().toLocaleDateString('en-IN', {
            day: 'numeric',
            month: 'long',
            year: 'numeric'
        });
    }

    const alertToggle = document.getElementById('alert-toggle');
    if (alertToggle) {
        alertToggle.addEventListener('click', () => {
            alertToggle.classList.toggle('on');
        });
    }

    // --- Live and Predicted Data Integration ---
    const lanePieCanvas = document.getElementById('lanePie');
    const legendEl = document.getElementById('legend');
    let lanePie = null;
    const labels = ['Left Lane', 'Right Lane'];
    const colors = ['#ff6b6b', '#4ade80'];

    const predictionCanvas = document.getElementById('predictionChart');
    let predictionChart = null;
    
    const leftSignalEl = document.getElementById('left-signal');
    const rightSignalEl = document.getElementById('right-signal');
    const priorityAlertEl = document.getElementById('priority-alert');
    const emergencyVehicleTypeEl = document.getElementById('emergency-vehicle-type');


    const updateCharts = (realtimeData, predictedData) => {
        // Update Pie Chart with Real-time Data
        const realTimeValues = [realtimeData.left_lane, realtimeData.right_lane];
        if (!lanePie) {
            if (lanePieCanvas) {
                const ctx = lanePieCanvas.getContext('2d');
                lanePie = new Chart(ctx, {
                    type: 'pie',
                    data: { labels: labels, datasets: [{ data: realTimeValues, backgroundColor: colors, borderColor: 'rgba(30, 30, 30, 0.4)', borderWidth: 4 }] },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: true } } }
                });
            }
        } else {
            lanePie.data.datasets[0].data = realTimeValues;
            lanePie.update();
        }

        if (legendEl) {
            legendEl.innerHTML = '';
            const total = realTimeValues[0] + realTimeValues[1];
            const buildLegendItem = (label, color, value) => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                const sw = document.createElement('div');
                sw.className = 'swatch';
                sw.style.background = color;
                const txt = document.createElement('div');
                const percentage = total > 0 ? Math.round(value / total * 100) : 0;
                txt.innerHTML = `<strong>${label}</strong> — <span class="text-gray-300">${value} (${percentage}%)</span>`;
                item.appendChild(sw);
                item.appendChild(txt);
                legendEl.appendChild(item);
            };
            buildLegendItem(labels[0], colors[0], realTimeValues[0]);
            buildLegendItem(labels[1], colors[1], realTimeValues[1]);
        }
        
        // Update Prediction Chart
        if (predictionCanvas) {
            const predLabels = ['Left Lane', 'Right Lane'];
            const predDataPoints = [predictedData.left_lane, predictedData.right_lane];
            const predColors = ['#ff6b6b', '#60a5fa'];
            
            if (!predictionChart) {
                const ctx = predictionCanvas.getContext('2d');
                predictionChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: predLabels,
                        datasets: [{
                            label: 'Predicted Vehicles',
                            data: predDataPoints,
                            backgroundColor: predColors,
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } }
                    }
                });
            } else {
                predictionChart.data.datasets[0].data = predDataPoints;
                predictionChart.update();
            }
        }
    };

    // WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateCharts(data.realtime, data.predicted);
        
        // Update signal colors
        if (data.signal) {
            if (data.signal.left_lane === "green") {
                leftSignalEl.classList.add("bg-green-500");
                leftSignalEl.classList.remove("bg-red-500");
            } else {
                leftSignalEl.classList.add("bg-red-500");
                leftSignalEl.classList.remove("bg-green-500");
            }
            if (data.signal.right_lane === "green") {
                rightSignalEl.classList.add("bg-green-500");
                rightSignalEl.classList.remove("bg-red-500");
            } else {
                rightSignalEl.classList.add("bg-red-500");
                rightSignalEl.classList.remove("bg-green-500");
            }
        }
        
        // Show/hide priority alert and update vehicle type
        if (data.priority_alert) {
            if (data.priority_alert.left_lane || data.priority_alert.right_lane) {
                priorityAlertEl.classList.remove('hidden');
                emergencyVehicleTypeEl.textContent = data.priority_vehicle_type;
            } else {
                priorityAlertEl.classList.add('hidden');
                emergencyVehicleTypeEl.textContent = "None";
            }
        }
    };
});