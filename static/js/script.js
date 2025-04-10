// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded and ready");

    // Elements
    const uploadTabs = document.querySelectorAll('.upload-tab');
    const textInputArea = document.getElementById('text-input');
    const fileUploadArea = document.getElementById('file-upload');
    const csvUploadArea = document.getElementById('csv-upload-area');
    const csvFileInput = document.getElementById('csv-file-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const abstractText = document.getElementById('abstract-text');
    const submitAbstractBtn = document.getElementById('submit-abstract-btn');
    const submitBtnText = submitAbstractBtn.querySelector('.btn-text');
    const submitBtnLoader = submitAbstractBtn.querySelector('.btn-loader');
    const seeTableBtn = document.getElementById('see-table-btn');
    const seeQaPairsBtn = document.getElementById('see-qa-pairs-btn');
    const abstractProcessingStatus = document.getElementById('abstract-processing-status');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendMessageBtn = document.getElementById('send-message');
    const clearChatBtn = document.getElementById('clear-chat');
    const charCount = document.getElementById('char-count');
    const tableModal = document.getElementById('table-modal');
    const tableContent = document.getElementById('table-content');
    const closeBtn = document.querySelector('.close');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const cleanCacheBtn = document.getElementById('clean-cache-btn');
    const toggleModelBtn = document.getElementById('toggle-model-btn');
    const themeToggleBtn = document.getElementById('theme-toggle');
    const progressContainer = document.querySelector('.upload-progress-container');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    const tableSearch = document.getElementById('table-search');
    const qaPairsModal = document.getElementById('qa-pairs-modal');
    const qaContent = document.getElementById('qa-content');
    const qaSearch = document.getElementById('qa-search');
    const qaCloseBtn = qaPairsModal ? qaPairsModal.querySelector('.close') : null;
    const downloadQaJsonBtn = document.getElementById('download-qa-json-btn');

    // Initialize variables
    let messagesExist = false;
    let currentSessionId = null;
    let abstractProcessed = false;
    let selectedFile = null;
    let uploadMode = 'text'; // Default to text input mode
    let usingApiModel = true;
    let tableData = [];
    let qaPairsData = [];
    let qaPairsStatusInterval = null;


    // Theme management
    function initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);

        // Update background animation colors
        updatePathColors(newTheme);

        // Show a toast notification
        showToast(`${newTheme.charAt(0).toUpperCase() + newTheme.slice(1)} theme activated`, 'info');
    }

    function updateThemeIcon(theme) {
        if (themeToggleBtn) {
            if (theme === 'dark') {
                themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
            } else {
                themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
            }
        }
    }

        // Background Animation
    // function initBackgroundAnimation() {
    //     const svg = document.querySelector('.background-paths');
    //     if (!svg) return;
        
    //     // Clear existing paths
    //     svg.innerHTML = '<title>Background Animation</title>';
        
    //     // Get current theme
    //     const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        
    //     // Create paths for both directions
    //     createPaths(svg, 1, currentTheme);
    //     createPaths(svg, -1, currentTheme);
        
    //     // Update paths when theme changes
    //     const observer = new MutationObserver((mutations) => {
    //         mutations.forEach((mutation) => {
    //             if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
    //                 const newTheme = document.documentElement.getAttribute('data-theme') || 'light';
    //                 updatePathColors(newTheme);
    //             }
    //         });
    //     });
        
    //     // Start observing
    //     observer.observe(document.documentElement, { attributes: true });
    // }

    // function createPaths(svg, position, theme) {
    //     // Generate fewer paths for subtlety (20 instead of 36)
    //     const pathCount = 50;

    //     // Get viewport dimensions for relative scaling
    //     const viewBox = "0 0 1920 1080";  // Default viewBox size
    //     svg.setAttribute('viewBox', viewBox);
    //     svg.setAttribute('preserveAspectRatio', 'none');  // This helps cover the full width
        
    //     for (let i = 0; i < pathCount; i++) {
    //         const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

    //         // Calculate vertical offset to spread lines across the screen height
    //         const vertOffset = i * 30 - 400;
            
    //         // Create wider, more spread-out paths
    //         const d = `M-${480 - i * 15 * position} ${vertOffset}C-${
    //             380 - i * 20 * position
    //         } ${vertOffset} ${
    //             400 - i * 15 * position
    //         } ${vertOffset + 300} ${
    //             1200 + i * 15 * position
    //         } ${vertOffset + 400}C${1600 + i * 20 * position} ${vertOffset + 500} ${
    //             2000 + i * 15 * position
    //         } ${vertOffset + 600} ${2000 + i * 15 * position} ${vertOffset + 600}`;
            
    //         // Set attributes
    //         path.setAttribute('d', d);
    //         path.setAttribute('class', 'animated-path');
    //         // path.setAttribute('stroke-width', (0.7 + i * 0.03).toString());
    //         path.setAttribute('stroke-width', (0.8 + i * 0.05).toString());

    //         // Much lower opacity for subtlety
    //         // const opacity = theme === 'light' ? 0.03 + i * 0.006 : 0.04 + i * 0.007;
    //         const opacity = theme === 'light' ? 0.06 + i * 0.009 : 0.08 + i * 0.01;
    //         path.setAttribute('stroke-opacity', opacity.toString());

    //         const strokeColor = theme === 'light' ? '#5E7CE2' : '#6D8AE8';
    //         path.setAttribute('stroke', strokeColor);
            
    //         // Add random animation delay for more natural movement
    //         path.style.animationDuration = (20 + Math.random() * 20) + 's';
    //         path.style.animationDelay = (Math.random() * 8) + 's';
            
    //         svg.appendChild(path);
    //     }
    // }

    // function updatePathColors(theme) {
    //     const paths = document.querySelectorAll('.animated-path');
    //     paths.forEach((path, i) => {
    //         // Update opacity based on theme
    //         // const opacity = theme === 'light' ? 0.02 + (i % 20) * 0.005 : 0.03 + (i % 20) * 0.005;
    //         const opacity = theme === 'light' ? 0.06 + (i % 30) * 0.009 : 0.08 + (i % 30) * 0.01;
    //         path.setAttribute('stroke-opacity', opacity.toString());
    //     });
    // }

    // // Initialize background animation
    // initBackgroundAnimation();

    // function initBackgroundAnimation() {
    //     const svg = document.querySelector('.background-paths');
    //     if (!svg) return;
    
    //     // Clear existing paths
    //     svg.innerHTML = '<title>Background Animation</title>';
    
    //     // Set a wide viewBox to allow for offscreen starting positions
    //     svg.setAttribute('viewBox', '-800 -400 2400 1200');
    //     svg.setAttribute('preserveAspectRatio', 'xMidYMid slice');
    
    //     // Get current theme
    //     const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        
    //     // Create lines flowing from edges
    //     createFlowingLines(svg, currentTheme);
    
    //     // Update paths when theme changes
    //     const observer = new MutationObserver((mutations) => {
    //         mutations.forEach((mutation) => {
    //             if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
    //                 const newTheme = document.documentElement.getAttribute('data-theme') || 'light';
    //                 updatePathColors(newTheme);
    //             }
    //         });
    //     });
    
    //     // Start observing
    //     observer.observe(document.documentElement, { attributes: true });
    // }
    
    // function createFlowingLines(svg, theme) {
    //     const lineCount = 40; // More lines for better coverage
    //     const halfCount = lineCount / 2;
        
    //     // Create lines for top half (left to right)
    //     for (let i = 0; i < halfCount; i++) {
    //         // Calculate vertical position - spread across top half
    //         const vertPos = -350 + (i * 700 / halfCount);
            
    //         // Create line path from left to right
    //         createLinePath(svg, theme, {
    //             startX: -800, // Start offscreen left
    //             startY: vertPos,
    //             endX: 1600,   // End offscreen right
    //             endY: vertPos + (Math.random() * 100 - 50), // Slight random vertical shift
    //             index: i,
    //             flowDirection: 'ltr', // left to right
    //             halfIndex: i
    //         });
    //     }
        
    //     // Create lines for bottom half (right to left)
    //     for (let i = 0; i < halfCount; i++) {
    //         // Calculate vertical position - spread across bottom half
    //         const vertPos = 0 + (i * 700 / halfCount);
            
    //         // Create line path from right to left
    //         createLinePath(svg, theme, {
    //             startX: 1600, // Start offscreen right
    //             startY: vertPos,
    //             endX: -800,   // End offscreen left
    //             endY: vertPos + (Math.random() * 100 - 50), // Slight random vertical shift
    //             index: i + halfCount,
    //             flowDirection: 'rtl', // right to left
    //             halfIndex: i
    //         });
    //     }
    // }
    
    // function createLinePath(svg, theme, options) {
    //     const { startX, startY, endX, endY, index, flowDirection, halfIndex } = options;
        
    //     // Create path element
    //     const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        
    //     // Generate a wavy path with control points
    //     const curve1X = startX + (endX - startX) * 0.33 + (Math.random() * 200 - 100);
    //     const curve1Y = startY + (Math.random() * 100 - 50);
    //     const curve2X = startX + (endX - startX) * 0.66 + (Math.random() * 200 - 100);
    //     const curve2Y = endY + (Math.random() * 100 - 50);
        
    //     const d = `M${startX},${startY} C${curve1X},${curve1Y} ${curve2X},${curve2Y} ${endX},${endY}`;
        
    //     // Set path attributes
    //     path.setAttribute('d', d);
    //     path.setAttribute('class', 'animated-path');
    //     path.setAttribute('stroke-width', (0.7 + halfIndex * 0.04).toString());
        
    //     // Higher opacity for more glow
    //     const opacity = theme === 'light' ? 
    //         0.05 + halfIndex * 0.007 : 
    //         0.07 + halfIndex * 0.009;
    //     path.setAttribute('stroke-opacity', opacity.toString());
        
    //     // Add custom animation with direction-based timing
    //     const baseSpeed = 25 + Math.random() * 20; // Base animation speed in seconds
    //     const delay = Math.random() * 15; // Random delay creates continuous flow
        
    //     // Custom animation class based on direction
    //     path.classList.add(`flow-${flowDirection}`);
    //     path.style.setProperty('--animation-duration', `${baseSpeed}s`);
    //     path.style.setProperty('--animation-delay', `-${delay}s`);
        
    //     svg.appendChild(path);
    // }
    
    // function updatePathColors(theme) {
    //     const paths = document.querySelectorAll('.animated-path');
    //     paths.forEach((path, i) => {
    //         // Determine if this is in first or second half
    //         const halfIndex = i % 20;
            
    //         // Update opacity based on theme with increased values for more glow
    //         const opacity = theme === 'light' ? 
    //             0.05 + halfIndex * 0.007 : 
    //             0.07 + halfIndex * 0.009;
    //         path.setAttribute('stroke-opacity', opacity.toString());
    //     });
    // }

    function initBackgroundAnimation() {
        const svg = document.querySelector('.background-paths');
        if (!svg) return;
    
        // Clear existing paths
        svg.innerHTML = '<title>Background Animation</title>';
        
        // Set a wider viewBox to allow for offscreen starting points
        svg.setAttribute('viewBox', '-800 -400 2400 1200');
        svg.setAttribute('preserveAspectRatio', 'xMidYMid slice');
    
        // Get current theme
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        
        // Create diagonal paths from both edges
        createDiagonalPaths(svg, 1, currentTheme);  // Left-to-right diagonal paths
        createDiagonalPaths(svg, -1, currentTheme); // Right-to-left diagonal paths
    
        // Update paths when theme changes (same as before)
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                    const newTheme = document.documentElement.getAttribute('data-theme') || 'light';
                    updatePathColors(newTheme);
                }
            });
        });
    
        observer.observe(document.documentElement, { attributes: true });
    }
    
    // function createDiagonalPaths(svg, position, theme) {
    //     // Generate paths for diagonal lines
    //     const pathCount = 20;
        
    //     for (let i = 0; i < pathCount; i++) {
    //         const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            
    //         // Adjust vertical spread
    //         const vertOffset = (i * 40) - 400;
            
    //         // Original diagonal path shape but extended to start from edges
    //         // If position is 1, start from left edge; if -1, start from right edge
    //         let startX, endX;
            
    //         if (position === 1) {
    //             // Left to right diagonal
    //             startX = -800;
    //             endX = 1600;
    //         } else {
    //             // Right to left diagonal
    //             startX = 1600;
    //             endX = -800;
    //         }
            
    //         // Create diagonal path
    //         const startY = vertOffset;
    //         const endY = vertOffset + 600; // Ensures diagonal angle
            
    //         // Control points for the curve - create a similar curve as original
    //         const ctrl1X = startX + (endX - startX) * 0.33;
    //         const ctrl1Y = startY + (endY - startY) * 0.2;
    //         const ctrl2X = startX + (endX - startX) * 0.66;
    //         const ctrl2Y = startY + (endY - startY) * 0.8;
            
    //         const d = `M${startX} ${startY} C${ctrl1X} ${ctrl1Y}, ${ctrl2X} ${ctrl2Y}, ${endX} ${endY}`;
            
    //         // Set attributes
    //         path.setAttribute('d', d);
    //         path.setAttribute('class', 'animated-path');
    //         path.setAttribute('stroke-width', (0.7 + i * 0.04).toString());
            
    //         // Higher opacity for more glow
    //         const opacity = theme === 'light' ? 
    //             0.05 + i * 0.007 : 
    //             0.07 + i * 0.009;
    //         path.setAttribute('stroke-opacity', opacity.toString());
            
    //         // Randomize animation
    //         const duration = (25 + Math.random() * 15) + 's';
    //         const delay = (Math.random() * 10) + 's';
    //         path.style.animationDuration = duration;
    //         path.style.animationDelay = delay;
            
    //         svg.appendChild(path);
    //     }
    // }
    
    // function updatePathColors(theme) {
    //     const paths = document.querySelectorAll('.animated-path');
    //     paths.forEach((path, i) => {
    //         // Determine path index within its group
    //         const pathIndex = i % 20;
            
    //         // Update opacity based on theme with increased values for more glow
    //         const opacity = theme === 'light' ? 
    //             0.05 + pathIndex * 0.007 : 
    //             0.07 + pathIndex * 0.009;
    //         path.setAttribute('stroke-opacity', opacity.toString());
    //     });
    // }
    
    function createDiagonalPaths(svg, position, theme) {
        // Generate paths for diagonal lines
        const pathCount = 20;
        
        for (let i = 0; i < pathCount; i++) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            
            // Adjust vertical spread
            const vertOffset = (i * 40) - 400;
            
            // Original diagonal path shape but extended to start from edges
            // If position is 1, start from left edge; if -1, start from right edge
            let startX, endX;
            
            if (position === 1) {
                // Left to right diagonal
                startX = -800;
                endX = 1600;
            } else {
                // Right to left diagonal
                startX = 1600;
                endX = -800;
            }
            
            // Create diagonal path
            const startY = vertOffset;
            const endY = vertOffset + 600; // Ensures diagonal angle
            
            // Control points for the curve - create a similar curve as original
            const ctrl1X = startX + (endX - startX) * 0.33;
            const ctrl1Y = startY + (endY - startY) * 0.2;
            const ctrl2X = startX + (endX - startX) * 0.66;
            const ctrl2Y = startY + (endY - startY) * 0.8;
            
            const d = `M${startX} ${startY} C${ctrl1X} ${ctrl1Y}, ${ctrl2X} ${ctrl2Y}, ${endX} ${endY}`;
            
            // Set attributes
            path.setAttribute('d', d);
            path.setAttribute('class', 'animated-path');
            
            // IMPROVED: Thicker lines for light mode
            const strokeWidth = theme === 'light' ? 
                (1.0 + i * 0.06) : // Thicker for light mode
                (0.7 + i * 0.04);  // Original for dark mode
            path.setAttribute('stroke-width', strokeWidth.toString());
            
            // IMPROVED: Much higher opacity for light mode
            const opacity = theme === 'light' ? 
                0.15 + i * 0.012 :  // Significantly higher for light mode
                0.07 + i * 0.009;   // Original for dark mode
            path.setAttribute('stroke-opacity', opacity.toString());
            
            // NEW: Set stroke color for better contrast in light mode
            const strokeColor = theme === 'light' ? 
                '#3F5BC4' :   // Darker blue for light mode (primary-dark from your CSS)
                'var(--text-color)'; // Default for dark mode
            path.setAttribute('stroke', strokeColor);
            
            // Randomize animation
            const duration = (25 + Math.random() * 15) + 's';
            const delay = (Math.random() * 10) + 's';
            path.style.animationDuration = duration;
            path.style.animationDelay = delay;
            
            svg.appendChild(path);
        }
    }
    
    function updatePathColors(theme) {
        const paths = document.querySelectorAll('.animated-path');
        paths.forEach((path, i) => {
            // Determine path index within its group
            const pathIndex = i % 20;
            
            // IMPROVED: Update opacity with higher values for light mode
            const opacity = theme === 'light' ? 
                0.15 + pathIndex * 0.012 :  // Significantly higher for light mode
                0.07 + pathIndex * 0.009;   // Original for dark mode
            path.setAttribute('stroke-opacity', opacity.toString());
            
            // IMPROVED: Update stroke width for light mode
            const strokeWidth = theme === 'light' ? 
                (1.0 + pathIndex * 0.06) : // Thicker for light mode
                (0.7 + pathIndex * 0.04);  // Original for dark mode
            path.setAttribute('stroke-width', strokeWidth.toString());
            
            // NEW: Update stroke color for better contrast in light mode
            const strokeColor = theme === 'light' ? 
                '#3F5BC4' :   // Darker blue for light mode (primary-dark from your CSS)
                'var(--text-color)'; // Default for dark mode
            path.setAttribute('stroke', strokeColor);
        });
    }
    
    
    initBackgroundAnimation();

    initializeTheme();

    // Initialize theme

    // Theme toggle button
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // Toast notification system
    function showToast(message, type = 'info', duration = 3000) {
        const toastContainer = document.getElementById('toast-container');

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        let icon;
        switch (type) {
            case 'success':
                icon = 'check-circle';
                break;
            case 'error':
                icon = 'exclamation-circle';
                break;
            case 'warning':
                icon = 'exclamation-triangle';
                break;
            default:
                icon = 'info-circle';
        }

        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;

        toastContainer.appendChild(toast);

        // Auto-remove toast after duration
        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, duration);
    }

    // Check initial model status and force update
    if (toggleModelBtn) {
        fetch('/api/get-model-status')
            .then(response => response.json())
            .then(data => {
                console.log("Initial model status:", data);
                usingApiModel = data.using_api;
                toggleModelBtn.innerHTML = `<i class="fas fa-exchange-alt"></i><span>Using: ${data.model_name}</span>`;

                // Set button color based on model
                if (usingApiModel) {
                    toggleModelBtn.style.backgroundColor = 'var(--info-color)';
                } else {
                    toggleModelBtn.style.backgroundColor = 'var(--warning-color)';
                }
            })
            .catch(error => {
                console.error("Error fetching model status:", error);
                showToast('Error checking model status', 'error');
                // Default to API if there's an error
                usingApiModel = true;
                toggleModelBtn.innerHTML = '<i class="fas fa-exchange-alt"></i><span>Using: API (Novita)</span>';
                toggleModelBtn.style.backgroundColor = 'var(--info-color)';
            });
    }

    // Toggle model button
    if (toggleModelBtn) {
        toggleModelBtn.addEventListener('click', function() {
            // Show a micro-interaction on button click
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);

            fetch('/api/toggle-model', {
                method: 'POST'
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        usingApiModel = data.using_api;
                        toggleModelBtn.innerHTML = `<i class="fas fa-exchange-alt"></i><span>Using: ${data.model_name}</span>`;

                        // Change button color based on model
                        if (usingApiModel) {
                            toggleModelBtn.style.backgroundColor = 'var(--info-color)';
                        } else {
                            toggleModelBtn.style.backgroundColor = 'var(--warning-color)';
                        }

                        // Show a message in the chat
                        const modelMessage = `Switched to ${data.model_name} model.`;
                        addSystemMessage(modelMessage);

                        // Show a toast notification
                        showToast(`Now using ${data.model_name}`, 'success');
                    } else {
                        showToast('Error toggling model: ' + data.error, 'error');
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                    showToast('Error: ' + error.message, 'error');
                });
        });
    }

    // Tab switching (Single/Multiple abstracts)
    uploadTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Add a ripple effect
            const ripple = document.createElement('span');
            ripple.classList.add('tab-ripple');
            this.appendChild(ripple);
            setTimeout(() => ripple.remove(), 500);

            // Update active tab
            uploadTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');

            // Show corresponding input area with smooth transition
            const targetId = this.getAttribute('data-target');
            if (targetId === 'text-input') {
                fileUploadArea.style.opacity = '0';
                setTimeout(() => {
                    textInputArea.style.display = 'block';
                    fileUploadArea.style.display = 'none';
                    setTimeout(() => {
                        textInputArea.style.opacity = '1';
                    }, 50);
                }, 300);
                uploadMode = 'text';
            } else {
                textInputArea.style.opacity = '0';
                setTimeout(() => {
                    textInputArea.style.display = 'none';
                    fileUploadArea.style.display = 'block';
                    setTimeout(() => {
                        fileUploadArea.style.opacity = '1';
                    }, 50);
                }, 300);
                uploadMode = 'file';
            }
            console.log("Changed to mode:", uploadMode);
        });
    });

    // CSV file upload
    if (csvUploadArea && csvFileInput) {
        csvUploadArea.addEventListener('click', function() {
            csvFileInput.click();
        });

        csvFileInput.addEventListener('change', function() {
            if (this.files.length) {
                selectedFile = this.files[0];
                fileNameDisplay.textContent = `Selected file: ${selectedFile.name}`;
                fileNameDisplay.style.color = 'var(--success-color)';

                // Show a toast notification
                showToast(`File selected: ${selectedFile.name}`, 'success');

                // Add animation
                csvUploadArea.style.borderColor = 'var(--success-color)';
                setTimeout(() => {
                    csvUploadArea.style.borderColor = '';
                }, 1000);
            }
        });

        // Drag and drop functionality
        csvUploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = 'var(--primary-color)';
            this.style.backgroundColor = 'rgba(94, 124, 226, 0.05)';
        });

        csvUploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '';
            this.style.backgroundColor = '';
        });

        csvUploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '';
            this.style.backgroundColor = '';

            if (e.dataTransfer.files.length) {
                const file = e.dataTransfer.files[0];
                if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                    selectedFile = file;
                    fileNameDisplay.textContent = `Selected file: ${file.name}`;
                    fileNameDisplay.style.color = 'var(--success-color)';

                    // Show animation
                    csvUploadArea.style.borderColor = 'var(--success-color)';
                    setTimeout(() => {
                        csvUploadArea.style.borderColor = '';
                    }, 1000);

                    // Show a toast notification
                    showToast(`File selected: ${file.name}`, 'success');
                } else {
                    showToast('Please upload a CSV file', 'error');
                }
            }
        });
    }

    // Character count
    if (chatInput) {
        chatInput.addEventListener('input', function() {
            const length = this.value.length;
            charCount.textContent = length;

            // Change color when approaching limit
            if (length > 900) {
                charCount.style.color = 'var(--warning-color)';
            } else if (length > 800) {
                charCount.style.color = 'var(--accent-color)';
            } else {
                charCount.style.color = '';
            }
        });
    }

    // Update progress bar
    function updateProgress(percent) {
        if (progressContainer && progressFill && progressText) {
            progressContainer.style.display = 'block';
            progressFill.style.width = `${percent}%`;
            progressText.textContent = `${percent}%`;

            if (percent >= 100) {
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 1000);
            }
        }
    }

    // QA Pairs Functionality
    function checkQaPairsStatus() {
        if (!currentSessionId) return;
        
        fetch(`/api/qa-pairs-status/${currentSessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed') {
                    // QA pairs are ready
                    seeQaPairsBtn.style.display = 'inline-block';
                    showToast('QA pairs generation completed', 'success');
                    // Stop checking
                    clearInterval(qaPairsStatusInterval);
                } else if (data.status === 'error') {
                    console.error('Error generating QA pairs:', data.error);
                    showToast('Error generating QA pairs', 'error');
                    // Stop checking
                    clearInterval(qaPairsStatusInterval);
                }
                // If still running, continue checking
            })
            .catch(error => {
                console.error('Error checking QA pairs status:', error);
                // Stop checking on error
                clearInterval(qaPairsStatusInterval);
            });
    }

    function loadQaPairs() {
        if (!currentSessionId) {
            showToast('No data available. Please process an abstract first.', 'warning');
            return;
        }
        
        // Show loading in the modal
        qaPairsModal.style.display = 'block';

            // Make sure the modal header has the download button
        // This ensures the button exists even if it wasn't in the original HTML
        const modalActions = qaPairsModal.querySelector('.modal-actions');
        if (modalActions && !document.getElementById('download-qa-json-btn')) {
            const downloadBtn = document.createElement('button');
            downloadBtn.id = 'download-qa-json-btn';
            downloadBtn.className = 'icon-btn';
            downloadBtn.innerHTML = '<i class="fas fa-download"></i><span>Download JSON</span>';
            modalActions.insertBefore(downloadBtn, qaPairsModal.querySelector('.close'));
            
            // Add event listener to the newly created button
            downloadBtn.addEventListener('click', function() {
                this.classList.add('btn-active');
                setTimeout(() => this.classList.remove('btn-active'), 200);
                
                // Show a toast notification
                showToast('Downloading QA pairs JSON...', 'info');
                
                // Trigger download
                window.location.href = `/api/download-qa-json/${currentSessionId}`;
            });
        }

        qaContent.innerHTML = '<div class="table-loading"><i class="fas fa-spinner fa-spin"></i> Loading QA pairs...</div>';
        
        fetch(`/api/get-qa-pairs/${currentSessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    qaPairsData = data.qa_pairs;
                    
                    if (qaPairsData.length === 0) {
                        qaContent.innerHTML = '<p>No QA pairs found.</p>';
                    } else {
                        renderQaPairsTable(qaPairsData);
                    }
                } else {
                    qaContent.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-circle"></i> Error: ${data.error}</p>`;
                    showToast('Error: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error("Error:", error);
                qaContent.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-circle"></i> Error loading QA pairs.</p>`;
                showToast('Error: ' + error.message, 'error');
            });
    }

    function renderQaPairsTable(data) {
        if (!data.length) {
            qaContent.innerHTML = '<p>No QA pairs found.</p>';
            return;
        }
        
        // Group data by Category for better organization
        const groupedData = {};
        data.forEach(item => {
            if (!groupedData[item.Category]) {
                groupedData[item.Category] = [];
            }
            groupedData[item.Category].push(item);
        });
        
        let html = '';
        
        // Create tabs for categories
        html += '<div class="qa-categories">';
        Object.keys(groupedData).forEach((category, index) => {
            html += `<div class="qa-category ${index === 0 ? 'active' : ''}" data-category="${category}">${category}</div>`;
        });
        html += '</div>';
        
        // Create content for each category
        html += '<div class="qa-content-container">';
        Object.keys(groupedData).forEach((category, index) => {
            html += `<div class="qa-category-content ${index === 0 ? 'active' : ''}" data-category="${category}">`;
            
            // Add table for this category
            html += '<table>';
            html += '<thead><tr><th>Alloy</th><th>Question</th><th>Answer</th></tr></thead>';
            html += '<tbody>';
            
            groupedData[category].forEach(item => {
                html += `<tr>
                    <td>${item.Alloy}</td>
                    <td>${item.Question}</td>
                    <td>${item.Answer}</td>
                </tr>`;
            });
            
            html += '</tbody></table>';
            html += '</div>';
        });
        html += '</div>';
        
        qaContent.innerHTML = html;
        
        // Add event listeners for category tabs
        document.querySelectorAll('.qa-category').forEach(tab => {
            tab.addEventListener('click', function() {
                const category = this.getAttribute('data-category');
                
                // Update active tab
                document.querySelectorAll('.qa-category').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                // Show corresponding content
                document.querySelectorAll('.qa-category-content').forEach(content => {
                    if (content.getAttribute('data-category') === category) {
                        content.classList.add('active');
                    } else {
                        content.classList.remove('active');
                    }
                });
            });
        });
    }

    function filterQaPairs(searchValue) {
        if (!qaPairsData.length) return;
        
        const filteredData = qaPairsData.filter(item => {
            return (
                item.Question.toLowerCase().includes(searchValue) || 
                item.Answer.toLowerCase().includes(searchValue) ||
                item.Alloy.toLowerCase().includes(searchValue) ||
                item.Category.toLowerCase().includes(searchValue)
            );
        });
        
        renderQaPairsTable(filteredData);
    }

    // Submit abstract
    if (submitAbstractBtn) {
        submitAbstractBtn.addEventListener('click', function() {
            console.log("Submit button clicked");

            // Add button animation
            submitBtnText.style.display = 'none';
            submitBtnLoader.style.display = 'inline-block';

            if (uploadMode === 'text') {
                // Process single abstract
                const abstract = abstractText.value.trim();
                if (!abstract) {
                    showToast('Please enter an abstract', 'warning');
                    submitBtnText.style.display = 'inline-block';
                    submitBtnLoader.style.display = 'none';
                    return;
                }

                // Show processing status
                showProcessingStatus('Processing abstract...', 'processing');

                submitAbstractBtn.disabled = true;

                // Send to backend
                fetch('/api/submit-abstract', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: abstract })
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log("Response:", data);

                        if (data.success) {
                            currentSessionId = data.session_id;
                            abstractProcessed = true;
                            
                            // Start checking QA pairs status if generation was started
                            if (data.qa_generation_started) {
                                // Clear any existing interval
                                if (qaPairsStatusInterval) {
                                    clearInterval(qaPairsStatusInterval);
                                }
                                
                                // Check status every 5 seconds
                                qaPairsStatusInterval = setInterval(checkQaPairsStatus, 5000);
                            }

                            showProcessingStatus('Abstract processed successfully! ', 'success');
                            seeTableBtn.style.display = 'inline-block';

                            // Update chat
                            addSystemMessage('Abstract processed successfully. I\'m ready to answer your questions about it! ');

                            // Show a toast notification
                            showToast('Abstract processed successfully! ', 'success');
                        } else {
                            showProcessingStatus('Error: ' + data.error, 'error');
                            showToast('Error: ' + data.error, 'error');
                        }
                    })
                    .catch(error => {
                        console.error("Error:", error);
                        showProcessingStatus('Error: ' + error.message, 'error');
                        showToast('Error: ' + error.message, 'error');
                    })
                    .finally(() => {
                        submitAbstractBtn.disabled = false;
                        submitBtnText.style.display = 'inline-block';
                        submitBtnLoader.style.display = 'none';
                    });

            } else {
                // Process CSV file
                if (!selectedFile) {
                    showToast('Please select a CSV file', 'warning');
                    submitBtnText.style.display = 'inline-block';
                    submitBtnLoader.style.display = 'none';
                    return;
                }

                // Show processing status
                showProcessingStatus('Processing CSV file...', 'processing');

                submitAbstractBtn.disabled = true;

                // Prepare form data
                const formData = new FormData();
                formData.append('file', selectedFile);

                // Simulate upload progress
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += 5;
                    if (progress > 90) {
                        clearInterval(progressInterval);
                    }
                    updateProgress(progress);
                }, 200);

                // Send to backend
                fetch('/api/submit-csv', {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log("Response:", data);
                        clearInterval(progressInterval);
                        updateProgress(100);

                        if (data.success) {
                            currentSessionId = data.session_id;
                            abstractProcessed = true;
                            
                            // Start checking QA pairs status if generation was started
                            if (data.qa_generation_started) {
                                // Clear any existing interval
                                if (qaPairsStatusInterval) {
                                    clearInterval(qaPairsStatusInterval);
                                }
                                
                                // Check status every 5 seconds
                                qaPairsStatusInterval = setInterval(checkQaPairsStatus, 5000);
                            }

                            showProcessingStatus('CSV processed successfully! ', 'success');
                            seeTableBtn.style.display = 'inline-block';

                            // Update chat
                            const messageText = `${data.count} abstracts processed successfully. I'm ready to answer your questions about them!`;
                            addSystemMessage(messageText);

                            // Show a toast notification
                            showToast(`Processed ${data.count} abstracts`, 'success');
                        } else {
                            showProcessingStatus('Error: ' + data.error, 'error');
                            showToast('Error: ' + data.error, 'error');
                        }
                    })
                    .catch(error => {
                        clearInterval(progressInterval);
                        console.error("Error:", error);
                        showProcessingStatus('Error: ' + error.message, 'error');
                        showToast('Error: ' + error.message, 'error');
                    })
                    .finally(() => {
                        submitAbstractBtn.disabled = false;
                        submitBtnText.style.display = 'inline-block';
                        submitBtnLoader.style.display = 'none';
                    });
            }
        });
    }

    // Helper function to show processing status
    function showProcessingStatus(message, className) {
        let icon = '';
        switch (className) {
            case 'processing':
                icon = '<i class="fas fa-spinner fa-spin"></i>';
                break;
            case 'success':
                icon = '<i class="fas fa-check-circle"></i>';
                break;
            case 'error':
                icon = '<i class="fas fa-exclamation-circle"></i>';
                break;
        }

        abstractProcessingStatus.innerHTML = `${icon} ${message}`;
        abstractProcessingStatus.style.display = 'flex';
        abstractProcessingStatus.className = className;
    }

    // Add system message to chat
    function addSystemMessage(text) {
        if (!messagesExist) {
            chatMessages.innerHTML = '';
            messagesExist = true;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        messageBubble.textContent = text;

        messageDiv.appendChild(messageBubble);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom with animation
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Add user or assistant message to chat
    function addMessage(text, type = 'assistant') {
        if (!messagesExist) {
            chatMessages.innerHTML = '';
            messagesExist = true;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = `message-avatar ${type}-avatar`;

        if (type === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }

        // Create message bubble
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';

        // Handle code blocks with syntax highlighting
        if (text.includes('```')) {
            const parts = text.split(/(```(?:[\w-]+)?\n[\s\S]*?\n```)/g);
            for (let part of parts) {
                if (part.startsWith('```') && part.endsWith('```')) {
                    // Extract language and code
                    const match = part.match(/```([\w-]+)?\n([\s\S]*?)\n```/);
                    if (match) {
                        const language = match[1] || '';
                        const code = match[2];

                        // Create code block elements
                        const codeBlock = document.createElement('pre');
                        const codeElement = document.createElement('code');
                        if (language) {
                            codeElement.className = `language-${language}`;
                        }
                        codeElement.textContent = code;
                        codeBlock.appendChild(codeElement);
                        messageBubble.appendChild(codeBlock);
                    }
                } else if (part.trim()) {
                    // Regular text
                    const textNode = document.createElement('p');
                    textNode.textContent = part;
                    messageBubble.appendChild(textNode);
                }
            }
        } else {
            // Regular text with paragraph formatting
            const paragraphs = text.split('\n\n');
            for (let paragraph of paragraphs) {
                if (paragraph.trim()) {
                    const p = document.createElement('p');
                    p.textContent = paragraph;
                    messageBubble.appendChild(p);
                }
            }
        }

        // Add avatar and message bubble to the message container
        if (type === 'user') {
            messageDiv.appendChild(messageBubble);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageBubble);
        }

        chatMessages.appendChild(messageDiv);

        // Add entrance animation class
        setTimeout(() => {
            messageDiv.classList.add('message-animate');
        }, 10);

        // Scroll to bottom with animation
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Add typing indicator
    function addTypingIndicator() {
        if (!messagesExist) {
            chatMessages.innerHTML = '';
            messagesExist = true;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.id = 'typing-indicator-container';

        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar assistant-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';

        // Create typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(typingIndicator);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom with animation
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Remove typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator-container');
        if (indicator) {
            indicator.classList.add('fade-out');
            setTimeout(() => {
                indicator.remove();
            }, 300);
        }
    }

    // Send message
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Check if abstract has been processed
        if (!abstractProcessed) {
            addSystemMessage('Please submit an abstract first before asking questions. ');
            showToast('Please submit an abstract first', 'warning');
            return;
        }

        // Add user message
        addMessage(message, 'user');

        // Clear input with animation
        chatInput.value = '';
        chatInput.style.height = 'auto';
        charCount.textContent = '0';
        charCount.style.color = '';

        // Show typing indicator
        addTypingIndicator();

        // Send to backend
        fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message })
        })
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                removeTypingIndicator();

                // Add assistant response with slight delay for natural feeling
                setTimeout(() => {
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        addMessage('I received an empty response. Please try a different question. ');
                    }
                }, 300);
            })
            .catch(error => {
                console.error("Error:", error);
                removeTypingIndicator();
                addSystemMessage(`Sorry, an error occurred. Please try again.`);
                showToast('Error: ' + error.message, 'error');
            });
    }

    // Send message button
    if (sendMessageBtn) {
        sendMessageBtn.addEventListener('click', function() {
            // Add button press animation
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);

            sendMessage();
        });
    }

    // Send message on Enter
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight < 150) ? this.scrollHeight + 'px' : '150px';
        });
    }

    // Clear chat
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', function() {
            // Add button animation
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);

            // Fade out chat messages
            chatMessages.style.opacity = '0';

            setTimeout(() => {
                chatMessages.innerHTML = `
                    <div class="empty-chat">
                        <i class="fas fa-comments fa-3x"></i>
                        <p>No messages yet. Start a conversation!</p>
                    </div>
                `;
                chatMessages.style.opacity = '1';
                messagesExist = false;
            }, 300);

            showToast('Chat cleared', 'info');
        });
    }

    // Table search functionality
    if (tableSearch) {
        tableSearch.addEventListener('input', function() {
            const searchValue = this.value.toLowerCase();
            filterTable(searchValue);
        });
    }

    // QA pairs search functionality
    if (qaSearch) {
        qaSearch.addEventListener('input', function() {
            const searchValue = this.value.toLowerCase();
            filterQaPairs(searchValue);
        });
    }

    function filterTable(searchValue) {
        if (!tableData.length) return;

        const filteredData = tableData.filter(row => {
            return Object.values(row).some(value =>
                String(value).toLowerCase().includes(searchValue)
            );
        });

        renderTable(filteredData);
    }

    function renderTable(data) {
        if (!data.length) {
            tableContent.innerHTML = '<p>No matching data found.</p>';
            return;
        }

            // Define the correct order of columns
        const columnOrder = [
            "Index", "DOI", "Alloy Name", "Property Name", "Property Type",
            "Temperature", "Value", "Unit", "Equilibrium Conditions",
            "Single/Multiphase", "Phase Type"
        ];

        // Create table
        // const headers = Object.keys(data[0]);

        let tableHtml = '<table>';

            // Headers with specified order
        tableHtml += '<thead><tr>';
        columnOrder.forEach(header => {
            tableHtml += `<th>${header}</th>`;
        });
        tableHtml += '</tr></thead>';

        // // Headers with sort functionality
        // tableHtml += '<thead><tr>';
        // headers.forEach(header => {
        //     tableHtml += `<th data-sort="${header}">${header} <i class="fas fa-sort"></i></th>`;
        // });
        // tableHtml += '</tr></thead>';

        // Data rows
        tableHtml += '<tbody>';
        data.forEach(row => {
            tableHtml += '<tr>';
            columnOrder.forEach(header => {
                const valueToDisplay = row[header] !== undefined ? row[header] : '';
                tableHtml += `<td>${valueToDisplay}</td>`;
                // tableHtml += `<td>${row[header] || ''}</td>`;
            });
            tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';

        tableContent.innerHTML = tableHtml;

        // Add sort event listeners
        document.querySelectorAll('[data-sort]').forEach(th => {
            th.addEventListener('click', function() {
                const header = this.getAttribute('data-sort');
                sortTable(header);

                // Toggle sort direction icon
                const direction = this.classList.contains('asc') ? 'desc' : 'asc';

                // Reset all headers
                document.querySelectorAll('[data-sort]').forEach(el => {
                    el.classList.remove('asc', 'desc');
                    el.querySelector('i').className = 'fas fa-sort';
                });

                // Set new direction
                this.classList.add(direction);
                this.querySelector('i').className = `fas fa-sort-${direction === 'asc' ? 'up' : 'down'}`;
            });
        });
    }

    function sortTable(header) {
        const th = document.querySelector(`[data-sort="${header}"]`);
        const isAscending = !th.classList.contains('asc');

        tableData.sort((a, b) => {
            const valueA = String(a[header] || '').toLowerCase();
            const valueB = String(b[header] || '').toLowerCase();

            // Try to sort as numbers if possible
            const numA = parseFloat(valueA);
            const numB = parseFloat(valueB);

            if (!isNaN(numA) && !isNaN(numB)) {
                return isAscending ? numA - numB : numB - numA;
            }

            // Otherwise sort as strings
            if (valueA < valueB) return isAscending ? -1 : 1;
            if (valueA > valueB) return isAscending ? 1 : -1;
            return 0;
        });

        renderTable(tableData);
    }

    // See table button
    if (seeTableBtn) {
        seeTableBtn.addEventListener('click', function() {
            if (!currentSessionId) {
                showToast('No data available. Please process an abstract first. ', 'warning');
                return;
            }

            // Add button animation
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);

            // Show loading in the modal
            tableModal.style.display = 'block';
            tableContent.innerHTML = '<div class="table-loading"><i class="fas fa-spinner fa-spin"></i> Loading data...</div>';

            fetch(`/api/get-table/${currentSessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        tableData = data.table_data;

                        if (tableData.length === 0) {
                            tableContent.innerHTML = '<p>No data found in the abstract.</p>';
                        } else {
                            renderTable(tableData);
                        }
                    } else {
                        tableContent.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-circle"></i> Error: ${data.error}</p>`;
                        showToast('Error: ' + data.error, 'error');
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                    tableContent.innerHTML = `<p class="error-message"><i class="fas fa-exclamation-circle"></i> Error loading data.</p>`;
                    showToast('Error: ' + error.message, 'error');
                });
        });
    }
    
    // See QA pairs button
    if (seeQaPairsBtn) {
        seeQaPairsBtn.addEventListener('click', function() {
            if (!currentSessionId) {
                showToast('No data available. Please process an abstract first.', 'warning');
                return;
            }
            
            // Add button animation
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);
            
            loadQaPairs();
        });
    }

    // Download JSON
    if (downloadJsonBtn) {
        downloadJsonBtn.addEventListener('click', function() {
            if (!currentSessionId) {
                showToast('No data available. Please process an abstract first. ', 'warning');
                return;
            }

            // Add button animation
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);

            // Show a toast notification
            showToast('Downloading JSON data...', 'info');

            window.location.href = `/api/download-json/${currentSessionId}`;
        });
    }

    // Close modals
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            tableModal.classList.add('modal-closing');
            setTimeout(() => {
                tableModal.style.display = 'none';
                tableModal.classList.remove('modal-closing');
            }, 300);
        });
    }
    
    if (qaCloseBtn) {
        qaCloseBtn.addEventListener('click', function() {
            qaPairsModal.classList.add('modal-closing');
            setTimeout(() => {
                qaPairsModal.style.display = 'none';
                qaPairsModal.classList.remove('modal-closing');
            }, 300);
        });
    }

    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === tableModal) {
            tableModal.classList.add('modal-closing');
            setTimeout(() => {
                tableModal.style.display = 'none';
                tableModal.classList.remove('modal-closing');
            }, 300);
        }
        
        if (event.target === qaPairsModal) {
            qaPairsModal.classList.add('modal-closing');
            setTimeout(() => {
                qaPairsModal.style.display = 'none';
                qaPairsModal.classList.remove('modal-closing');
            }, 300);
        }
    });

    // Clean cache
    if (cleanCacheBtn) {
        cleanCacheBtn.addEventListener('click', function() {
            // Add button animation
            this.classList.add('btn-active');
            setTimeout(() => this.classList.remove('btn-active'), 200);

            // Show confirmation dialog
            if (confirm('This will clear model cache and free up disk space. Continue?')) {
                // Show processing state
                this.disabled = true;
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Cleaning...';

                fetch('/api/clean-cache', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-broom"></i><span>Clean Cache</span>';

                        if (data.success) {
                            showToast('Cache cleared successfully', 'success');
                        } else {
                            showToast('Error: ' + data.error, 'error');
                        }
                    })
                    .catch(error => {
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-broom"></i><span>Clean Cache</span>';
                        showToast('Error: ' + error.message, 'error');
                    });
            }
        });
    }
});
