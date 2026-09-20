document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const dropContent = document.getElementById('drop-content');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const statusBadge = document.getElementById('status-badge');
    const telemetryDevice = document.getElementById('telemetry-device');
    
    // States
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsState = document.getElementById('results-state');
    const reportDocInfo = document.getElementById('report-doc-info');
    
    let selectedFile = null;

    // Check API Health and Engine Telemetry
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ok') {
                statusBadge.textContent = 'SYSTEM ONLINE';
                statusBadge.className = 'text-xs bg-brand-accent text-brand-dark px-2 py-0.5 font-mono uppercase font-bold';
                if (telemetryDevice) {
                    telemetryDevice.textContent = `DEVICE: ${data.device.toUpperCase()} | MODEL: ${data.model_type.toUpperCase()}`;
                }
            }
        })
        .catch(err => {
            console.error('API Offline', err);
            statusBadge.textContent = 'SYSTEM OFFLINE';
            statusBadge.className = 'text-xs bg-brand-red text-white px-2 py-0.5 font-mono uppercase font-bold';
        });

    // Drag and Drop Handlers
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-brand-blue');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-brand-blue');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-brand-blue');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    if (resetBtn) {
        resetBtn.addEventListener('click', resetUI);
    }

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewImage.classList.add('hidden');
        dropContent.classList.remove('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.add('hidden');
        emptyState.classList.remove('hidden');
        if (resetBtn) resetBtn.classList.add('hidden');
        if (reportDocInfo) reportDocInfo.textContent = 'STANDBY';
    }

    function handleFile(file) {
        selectedFile = file;
        const isPdf = file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';
        
        if (resetBtn) resetBtn.classList.remove('hidden');

        if (isPdf) {
            // PDF document preview placeholder
            previewImage.classList.add('hidden');
            dropContent.classList.remove('hidden');
            document.getElementById('drop-text').textContent = `Loaded PDF: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        } else {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.classList.remove('hidden');
                dropContent.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }

        if (reportDocInfo) {
            reportDocInfo.textContent = `TARGET: ${file.name.toUpperCase()}`;
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('PLEASE SELECT OR DROP A DOCUMENT FIRST.');
            return;
        }

        // UI Transition to Loading
        emptyState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span>AUDITING DOCUMENT...</span>';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/v1/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Analysis Failed (${response.status})`);
            }

            const result = await response.json();
            
            // 1. Render Verdict with Calibrated 3-State Styling
            const verdictEl = document.getElementById('result-verdict');
            const scoreEl = document.getElementById('result-score');
            const confidenceEl = document.getElementById('result-confidence');
            
            scoreEl.textContent = result.tamper_score.toFixed(3);
            confidenceEl.textContent = `${(result.confidence * 100).toFixed(1)}%`;
            
            if (result.verdict === 'TAMPERED') {
                verdictEl.textContent = 'TAMPERED';
                verdictEl.className = 'text-3xl font-display font-bold text-brand-red';
            } else if (result.verdict === 'UNCERTAIN') {
                verdictEl.textContent = 'UNCERTAIN';
                verdictEl.className = 'text-3xl font-display font-bold text-brand-amber';
            } else {
                verdictEl.textContent = 'AUTHENTIC';
                verdictEl.className = 'text-3xl font-display font-bold text-brand-accent';
            }

            // 2. Render Numerical Metrics
            document.getElementById('result-regions').textContent = result.regions_detected;
            if (result.forensics) {
                document.getElementById('result-ela-delta').textContent = result.forensics.ela_max_diff ?? '--';
                document.getElementById('result-noise-var').textContent = result.forensics.noise_residual_variance ? result.forensics.noise_residual_variance.toFixed(3) : '--';
            }

            // 3. Render Flagged Spliced Text Tokens
            const flaggedContainer = document.getElementById('flagged-text-container');
            const flaggedList = document.getElementById('flagged-text-list');
            if (result.flagged_text && result.flagged_text.length > 0) {
                flaggedList.innerHTML = result.flagged_text.map(t => 
                    `<span class="bg-black/40 text-brand-amber px-2 py-0.5 rounded border border-brand-amber/30">"${t.text}"</span>`
                ).join(' ');
                flaggedContainer.classList.remove('hidden');
            } else {
                flaggedContainer.classList.add('hidden');
            }

            // 4. Render Multimodal Report
            const vlmText = result.vlm_summary || 'No detailed forensic narrative generated.';
            document.getElementById('result-explanation').innerHTML = typeof marked !== 'undefined' ? marked.parse(vlmText) : vlmText;
            
            const shortSummaryEl = document.getElementById('vlm-short-text');
            if (result.verdict === 'TAMPERED') {
                shortSummaryEl.textContent = `High risk detected. Localized ${result.regions_detected} anomaly cluster(s) with anomalous compression variance.`;
            } else if (result.verdict === 'UNCERTAIN') {
                shortSummaryEl.textContent = `Borderline forensic metrics detected. Manual operator verification recommended.`;
            } else {
                shortSummaryEl.textContent = `Statistical compression and noise variance are consistent with an authentic document.`;
            }

            // 5. Render Evidence Artifacts (Base64 JPEG)
            if (result.artifacts) {
                if (result.artifacts.ela) document.getElementById('img-ela').src = 'data:image/jpeg;base64,' + result.artifacts.ela;
                if (result.artifacts.residual) document.getElementById('img-res').src = 'data:image/jpeg;base64,' + result.artifacts.residual;
                if (result.artifacts.mask) document.getElementById('img-mask').src = 'data:image/jpeg;base64,' + result.artifacts.mask;
                if (result.artifacts.annotated) {
                    document.getElementById('img-annotated').src = 'data:image/jpeg;base64,' + result.artifacts.annotated;
                    // Update main preview canvas to show annotated bounding boxes
                    previewImage.src = 'data:image/jpeg;base64,' + result.artifacts.annotated;
                    previewImage.classList.remove('hidden');
                    dropContent.classList.add('hidden');
                }
            }

            // 6. Update Document Info Header
            if (result.document_info && reportDocInfo) {
                const docType = result.document_info.is_pdf ? 'PDF' : 'IMAGE';
                reportDocInfo.textContent = `${docType} [${result.document_info.width}x${result.document_info.height}] - ${result.model.name}`;
            }

            loadingState.classList.add('hidden');
            resultsState.classList.remove('hidden');
            resultsState.classList.add('flex');
            
        } catch (error) {
            console.error(error);
            alert('FORENSIC AUDIT FAILED: ' + error.message);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<span>Execute Forensic Audit</span><span class="text-brand-accent">→</span>';
        }
    });
});
