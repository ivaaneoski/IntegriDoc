document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const dropText = document.getElementById('drop-text');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusBadge = document.getElementById('status-badge');
    
    // States
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsState = document.getElementById('results-state');
    
    let selectedFile = null;

    // Check API Health
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ok') {
                statusBadge.textContent = 'SYSTEM ONLINE';
                statusBadge.classList.replace('bg-brand-acid', 'bg-white');
            }
        })
        .catch(err => console.error('API Offline', err));

    // Drag and Drop Handlers
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-brand-pink');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-brand-pink');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-brand-pink');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            dropText.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('PLEASE UPLOAD A DOCUMENT FIRST.');
            return;
        }

        // UI Updates
        emptyState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'PROCESSING...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Send to FastAPI
            const response = await fetch('/v1/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis Failed');

            const result = await response.json();
            
            // Render Results
            document.getElementById('result-score').textContent = result.score.toFixed(4);
            
            const verdictEl = document.getElementById('result-verdict');
            if (result.score > 0.5) {
                verdictEl.textContent = 'TAMPERED';
                verdictEl.classList.replace('text-brand-black', 'text-red-600');
            } else {
                verdictEl.textContent = 'REAL';
                verdictEl.classList.replace('text-red-600', 'text-green-600');
            }

            document.getElementById('result-regions').textContent = `${result.regions_detected} Detected`;
            
            const vlmText = result.vlm_summary || 'No VLM output available.';
            document.getElementById('result-explanation').innerHTML = marked.parse(vlmText);

            // Render Artifacts (Assuming API returns base64 or static URLs)
            if (result.artifacts) {
                if (result.artifacts.ela) document.getElementById('img-ela').src = 'data:image/jpeg;base64,' + result.artifacts.ela;
                if (result.artifacts.residual) document.getElementById('img-res').src = 'data:image/jpeg;base64,' + result.artifacts.residual;
                if (result.artifacts.mask) document.getElementById('img-mask').src = 'data:image/jpeg;base64,' + result.artifacts.mask;
                if (result.artifacts.annotated) previewImage.src = 'data:image/jpeg;base64,' + result.artifacts.annotated;
            }

            loadingState.classList.add('hidden');
            resultsState.classList.remove('hidden');
            resultsState.classList.add('flex');
            
        } catch (error) {
            console.error(error);
            alert('SYSTEM ERROR: ' + error.message);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'EXECUTE ANALYSIS';
        }
    });
});
