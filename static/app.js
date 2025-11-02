class NASInterface {
    constructor() {
        this.statusInterval = null;
        this.isSearchRunning = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.showTab('logs');
        this.updateStatus();
    }

    bindEvents() {
        document.getElementById('start-btn').addEventListener('click', () => this.startSearch());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopSearch());
        document.getElementById('export-btn').addEventListener('click', () => this.exportResults());
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.showTab(tabName);
            });
        });

        // Real-time input validation
        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', () => this.validateInput(input));
        });
    }

    validateInput(input) {
        const value = parseInt(input.value);
        const min = parseInt(input.min);
        const max = parseInt(input.max);
        
        if (value < min || value > max) {
            input.style.borderColor = '#dc2626';
            input.style.boxShadow = '0 0 0 3px rgba(220, 38, 38, 0.1)';
        } else {
            input.style.borderColor = '#e1e8ed';
            input.style.boxShadow = 'none';
        }
    }

    getConfig() {
        return {
            train_samples: parseInt(document.getElementById('train_samples').value),
            val_samples: parseInt(document.getElementById('val_samples').value),
            population_size: parseInt(document.getElementById('population_size').value),
            max_generations: parseInt(document.getElementById('max_generations').value),
            mutation_rate: parseFloat(document.getElementById('mutation_rate').value),
            epochs_per_eval: parseInt(document.getElementById('epochs_per_eval').value),
            batch_size: parseInt(document.getElementById('batch_size').value),
            early_stopping_patience: parseInt(document.getElementById('early_stopping_patience').value)
        };
    }

    async startSearch() {
        if (this.isSearchRunning) return;

        const config = this.getConfig();
        
        // Validate configuration
        if (!this.validateConfig(config)) return;

        try {
            const response = await fetch('/api/start_search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const data = await response.json();

            if (data.error) {
                this.showNotification(data.error, 'error');
                return;
            }

            this.isSearchRunning = true;
            this.updateButtonStates();
            this.startStatusUpdates();
            this.showNotification('Search started successfully!', 'success');

        } catch (error) {
            this.showNotification(`Failed to start search: ${error.message}`, 'error');
        }
    }

    async stopSearch() {
        try {
            await fetch('/api/stop_search', { method: 'POST' });
            this.isSearchRunning = false;
            this.updateButtonStates();
            this.stopStatusUpdates();
            this.showNotification('Search stopped', 'warning');
        } catch (error) {
            this.showNotification(`Failed to stop search: ${error.message}`, 'error');
        }
    }

    validateConfig(config) {
        const errors = [];

        if (config.train_samples < 100 || config.train_samples > 50000) {
            errors.push('Training samples must be between 100 and 50,000');
        }
        if (config.val_samples < 50 || config.val_samples > 10000) {
            errors.push('Validation samples must be between 50 and 10,000');
        }
        if (config.population_size < 2 || config.population_size > 20) {
            errors.push('Population size must be between 2 and 20');
        }
        if (config.max_generations < 1 || config.max_generations > 50) {
            errors.push('Max generations must be between 1 and 50');
        }
        if (config.mutation_rate < 0 || config.mutation_rate > 1) {
            errors.push('Mutation rate must be between 0 and 1');
        }

        if (errors.length > 0) {
            this.showNotification(errors.join('\n'), 'error');
            return false;
        }

        return true;
    }

    updateButtonStates() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const exportBtn = document.getElementById('export-btn');

        if (this.isSearchRunning) {
            startBtn.disabled = true;
            startBtn.innerHTML = '<div class="loading-spinner"></div> Running...';
            stopBtn.disabled = false;
            exportBtn.disabled = true;
        } else {
            startBtn.disabled = false;
            startBtn.innerHTML = '🚀 Start Search';
            stopBtn.disabled = true;
            exportBtn.disabled = false;
        }
    }

    startStatusUpdates() {
        this.statusInterval = setInterval(() => this.updateStatus(), 2000);
    }

    stopStatusUpdates() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            // Update status display
            document.getElementById('status-text').textContent = data.status;
            
            // Update progress bar
            const progressFill = document.getElementById('progress-fill');
            const progress = data.progress || 0;
            progressFill.style.width = `${progress}%`;

            // Update logs
            this.updateLogs(data.logs || []);

            // Update results if available
            if (data.results && data.results.best_architecture) {
                this.updateResults(data.results);
                this.loadVisualization();
            }

            // Check if search completed
            if (!data.is_running && this.isSearchRunning) {
                this.isSearchRunning = false;
                this.updateButtonStates();
                this.stopStatusUpdates();
                
                if (data.results) {
                    this.showNotification('Search completed successfully!', 'success');
                } else if (data.status.includes('Error')) {
                    this.showNotification('Search failed. Check logs for details.', 'error');
                }
            }

        } catch (error) {
            console.error('Status update failed:', error);
        }
    }

    updateLogs(logs) {
        const logContainer = document.getElementById('log-container');
        
        logContainer.innerHTML = logs.map(log => {
            let className = 'log-entry';
            if (log.includes('Error') || log.includes('failed')) className += ' error';
            else if (log.includes('completed') || log.includes('success')) className += ' success';
            else if (log.includes('Warning') || log.includes('stopped')) className += ' warning';
            
            return `<div class="${className}">${this.escapeHtml(log)}</div>`;
        }).join('');
        
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    updateResults(results) {
        const architectureDisplay = document.getElementById('architecture-display');
        architectureDisplay.textContent = JSON.stringify(results.best_architecture, null, 2);
    }

    async loadVisualization() {
        try {
            const response = await fetch('/api/plot');
            const data = await response.json();
            
            if (data.plot) {
                const vizContainer = document.getElementById('visualization-container');
                vizContainer.innerHTML = `<img src="data:image/png;base64,${data.plot}" alt="Search Results Visualization">`;
            }
        } catch (error) {
            console.error('Failed to load visualization:', error);
        }
    }

    async exportResults() {
        try {
            const response = await fetch('/api/export', { method: 'POST' });
            const data = await response.json();
            
            if (data.message) {
                this.showNotification(data.message, 'success');
            } else if (data.error) {
                this.showNotification(data.error, 'error');
            }
        } catch (error) {
            this.showNotification(`Export failed: ${error.message}`, 'error');
        }
    }

    showTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });

        // Show selected tab content
        document.getElementById(`${tabName}-content`).classList.add('active');
        
        // Add active class to selected tab
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            max-width: 400px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;

        // Set background color based on type
        const colors = {
            success: '#059669',
            error: '#dc2626',
            warning: '#d97706',
            info: '#2563eb'
        };
        notification.style.background = colors[type] || colors.info;
        
        notification.textContent = message;
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new NASInterface();
});