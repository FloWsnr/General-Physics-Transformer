/**
 * GIF Zoom Functionality
 * Adds click-to-zoom capability for GIF images in the grid
 */

class GifZoom {
    constructor() {
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupZoomHandlers();
        });
    }

    setupZoomHandlers() {
        const images = document.querySelectorAll('.gif-grid-3x3-nospace img, .gif-grid-4x2-nospace img, .gif-grid-3x3 img, .gif-grid-2x2 img');

        images.forEach(img => {
            img.style.cursor = 'pointer';
            img.addEventListener('click', (e) => this.handleImageClick(e));
        });
    }

    handleImageClick(event) {
        const img = event.target;

        if (img.classList.contains('zoomed')) {
            this.closeZoom(img);
        } else {
            this.openZoom(img);
        }
    }

    openZoom(img) {
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'zoom-overlay';
        overlay.addEventListener('click', () => this.closeZoom(img));

        // Prevent wheel/zoom events on overlay
        overlay.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, { passive: false });

        // Create zoomed image container
        const zoomedContainer = document.createElement('div');
        zoomedContainer.className = 'zoomed-container';

        // Clone the image for zooming
        const zoomedImg = img.cloneNode(true);
        zoomedImg.className = 'zoomed-image';

        // Prevent zoom on the image itself
        zoomedImg.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, { passive: false });

        // Create close button
        const closeBtn = document.createElement('button');
        closeBtn.className = 'zoom-close';
        closeBtn.innerHTML = '×';
        closeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.closeZoom(img);
        });

        zoomedContainer.appendChild(zoomedImg);
        zoomedContainer.appendChild(closeBtn);
        overlay.appendChild(zoomedContainer);

        document.body.appendChild(overlay);
        img.classList.add('zoomed');

        // Prevent body scroll and zoom
        document.body.style.overflow = 'hidden';
        document.addEventListener('wheel', this.preventZoom, { passive: false });
        document.addEventListener('keydown', this.preventKeyboardZoom, { passive: false });

        // Store references for cleanup
        overlay._preventZoom = this.preventZoom;
        overlay._preventKeyboardZoom = this.preventKeyboardZoom;

        // Animate in
        requestAnimationFrame(() => {
            overlay.classList.add('active');
        });
    }

    closeZoom(img) {
        const overlay = document.querySelector('.zoom-overlay');
        if (overlay) {
            // Remove event listeners
            if (overlay._preventZoom) {
                document.removeEventListener('wheel', overlay._preventZoom);
            }
            if (overlay._preventKeyboardZoom) {
                document.removeEventListener('keydown', overlay._preventKeyboardZoom);
            }

            overlay.classList.remove('active');
            setTimeout(() => {
                overlay.remove();
                document.body.style.overflow = '';
            }, 300);
        }
        img.classList.remove('zoomed');
    }

    preventZoom(e) {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            e.stopPropagation();
        }
    }

    preventKeyboardZoom(e) {
        // Prevent Ctrl/Cmd + Plus/Minus/0 (zoom shortcuts)
        if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '-' || e.key === '0')) {
            e.preventDefault();
            e.stopPropagation();
        }
    }
}

// Initialize when script loads
new GifZoom();