// app.js

document.addEventListener('DOMContentLoaded', () => {
    // Set current date
    const dateElement = document.getElementById('current-date');
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    dateElement.textContent = new Date().toLocaleDateString('en-US', options);

    // Sync button logic
    const syncBtn = document.getElementById('sync-watch-btn');
    const predictionOutput = document.getElementById('prediction-output');
    const predictionInsight = document.getElementById('prediction-insight');
    const fatigueScoreDisplay = document.getElementById('fatigue-score');
    const predictionRing = document.querySelector('.prediction-ring');
    
    // Metric nodes
    const valSleep = document.getElementById('val-sleep');
    const valHrv = document.getElementById('val-hrv');
    const valSpo2 = document.getElementById('val-spo2');
    const valHr = document.getElementById('val-hr');
    const primaryFactor = document.getElementById('primary-factor');

    // Simulate backend model inference
    syncBtn.addEventListener('click', () => {
        // UI Loading State
        syncBtn.classList.add('syncing');
        syncBtn.innerHTML = '<span class="sync-icon">⟳</span> Syncing...';
        predictionOutput.textContent = 'Running model inference...';
        predictionOutput.classList.add('loading');
        predictionInsight.textContent = 'Fetching latest health metrics and passing through Random Forest pipeline.';
        
        // Reset metrics
        ['val-sleep', 'val-hrv', 'val-spo2', 'val-hr'].forEach(id => document.getElementById(id).textContent = '--');
        fatigueScoreDisplay.textContent = '--';
        predictionRing.style.background = 'conic-gradient(var(--glass-bg) 100%, var(--glass-bg) 0)';

        // Simulate async model delay (e.g. 1.5 seconds)
        setTimeout(() => {
            syncBtn.classList.remove('syncing');
            syncBtn.innerHTML = '<span class="sync-icon">⟳</span> Sync Watch Data';
            
            // Mock Model Data Results
            const modelResult = {
                predictionClass: 1, // 0 = Not Fatigued, 1 = Fatigued
                probabilityStr: "0.82",
                metrics: {
                    sleep: "5.2", // hrs
                    hrv: "32", // ms
                    spo2: "96", // %
                    restingHr: "65" // bpm
                },
                primaryInsights: "Low Sleep Duration (5.2h) triggered high fatigue probability."
            };

            // Update UI based on mocked prediction results
            predictionOutput.classList.remove('loading');
            
            if (modelResult.predictionClass === 1) {
                // Fatigued styling
                predictionOutput.textContent = 'High Fatigue Expected';
                predictionOutput.style.background = 'linear-gradient(to right, #EF4444, #FCA5A5)';
                predictionOutput.style.webkitBackgroundClip = 'text';
                predictionOutput.style.webkitTextFillColor = 'transparent';
                
                // Ring animation for bad result
                predictionRing.style.background = `conic-gradient(var(--accent-red) ${parseFloat(modelResult.probabilityStr) * 100}%, var(--glass-bg) 0)`;
                fatigueScoreDisplay.textContent = Math.round(parseFloat(modelResult.probabilityStr) * 100) + '%';
            } else {
                // Not fatigued styling
                predictionOutput.textContent = 'Fully Rested';
                predictionOutput.style.background = 'linear-gradient(to right, #10B981, #6EE7B7)';
                predictionOutput.style.webkitBackgroundClip = 'text';
                predictionOutput.style.webkitTextFillColor = 'transparent';
                
                // Ring animation for good result
                predictionRing.style.background = `conic-gradient(var(--accent-green) ${(1.0 - parseFloat(modelResult.probabilityStr)) * 100}%, var(--glass-bg) 0)`;
                fatigueScoreDisplay.textContent = Math.round((1.0 - parseFloat(modelResult.probabilityStr)) * 100) + '%';
            }

            predictionInsight.textContent = modelResult.primaryInsights;

            // Animate counting numbers in metrics
            animateValue(valSleep, 0, parseFloat(modelResult.metrics.sleep), 800, ' hrs');
            animateValue(valHrv, 0, parseInt(modelResult.metrics.hrv), 800, ' ms');
            animateValue(valSpo2, 80, parseInt(modelResult.metrics.spo2), 800, ' %');
            animateValue(valHr, 40, parseInt(modelResult.metrics.restingHr), 800, ' bpm');
            
            primaryFactor.textContent = 'Sleep Duration < 6h';
            primaryFactor.style.color = varColor('--accent-red');

        }, 1500);
    });

    // Utility for fetching CSS variables
    function varColor(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    // Number counting animation utility
    function animateValue(obj, start, end, duration, suffix = '') {
        let startTimestamp = null;
        const isFloat = end % 1 !== 0;
        
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            
            let currentVal = progress * (end - start) + start;
            if (isFloat) {
                obj.textContent = currentVal.toFixed(1) + suffix;
            } else {
                obj.textContent = Math.floor(currentVal) + suffix;
            }
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.textContent = end + suffix;
            }
        };
        window.requestAnimationFrame(step);
    }
});
