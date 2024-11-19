document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.querySelector('#referral-input');
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('#submit-btn');
    const processingStatus = document.querySelector('#processing-status');
    
    if (uploadArea && fileInput && form) {
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length) {
                fileInput.files = files;
                updateFileName(files[0].name);
            }
        });
        
        // File input change handler
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                updateFileName(e.target.files[0].name);
            }
        });
    }
    
    // Update file name display
    function updateFileName(filename) {
        const fileNameDisplay = document.querySelector('.file-name');
        if (fileNameDisplay) {
        // Form submission handler
        form.addEventListener('submit', (e) => {
            if (!fileInput.files.length) {
                e.preventDefault();
                alert('Please select a file first.');
                return;
            }
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.querySelector('.button-text').classList.add('d-none');
            submitBtn.querySelector('.spinner-border').classList.remove('d-none');
            processingStatus.classList.remove('d-none');
        });
            fileNameDisplay.textContent = filename;
        }
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
