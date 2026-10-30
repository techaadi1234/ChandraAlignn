/**
 * ChandraAlign - 10x10 ANMS (Adaptive Non-Maximal Suppression) Heatmap
 * Visualizes spatial distribution uniformity across 100 lunar sub-swath sectors.
 */

class ANMSGridHeatmap {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.options = Object.assign({
      gridSize: 10,
      onCellSelect: null,
      onCellHover: null
    }, options);

    this.selectedCell = null;
    this.gridData = [];
    this.initDOM();
  }

  initDOM() {
    if (!this.container) return;
    this.container.innerHTML = '';
    this.container.className = 'anms-grid-10x10';
  }

  /**
   * Load and render 10x10 2D matrix
   * @param {number[][]} matrix - 10x10 array of keypoint counts
   */
  render(matrix) {
    if (!this.container || !matrix || !matrix.length) return;
    this.gridData = matrix;
    this.container.innerHTML = '';

    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let r = 0; r < matrix.length; r++) {
      for (let c = 0; c < matrix[r].length; c++) {
        const val = matrix[r][c];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        const count = matrix[r] ? matrix[r][c] || 0 : 0;
        const cell = document.createElement('div');
        cell.className = 'anms-cell';
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.dataset.count = count;
        cell.textContent = count;

        // Apply color ramp based on count
        cell.style.backgroundColor = this.getColorForCount(count, minVal, maxVal);

        // Tooltip description
        cell.title = `Sector [${r+1}, ${c+1}]: ${count} ANMS keypoints (Spatial coverage)`;

        cell.addEventListener('mouseenter', () => {
          if (typeof this.options.onCellHover === 'function') {
            this.options.onCellHover({ row: r, col: c, count: count });
          }
        });

        cell.addEventListener('click', () => {
          this.selectCell(cell, r, c, count);
        });

        this.container.appendChild(cell);
      }
    }
  }

  getColorForCount(val, min, max) {
    // 0 = dark space blue, optimal 5-6 = emerald/cyan, 7+ = amber
    if (val <= 3) {
      return 'rgba(30, 41, 59, 0.85)';
    } else if (val === 4) {
      return 'rgba(2, 132, 199, 0.75)';
    } else if (val === 5) {
      return 'rgba(0, 240, 255, 0.65)';
    } else if (val === 6) {
      return 'rgba(0, 230, 118, 0.75)';
    } else {
      return 'rgba(255, 179, 0, 0.85)';
    }
  }

  selectCell(element, row, col, count) {
    const prev = this.container.querySelector('.anms-cell.selected');
    if (prev) prev.classList.remove('selected');

    if (this.selectedCell && this.selectedCell.row === row && this.selectedCell.col === col) {
      this.selectedCell = null;
      if (typeof this.options.onCellSelect === 'function') {
        this.options.onCellSelect(null);
      }
      return;
    }

    element.classList.add('selected');
    this.selectedCell = { row, col, count };

    if (typeof this.options.onCellSelect === 'function') {
      this.options.onCellSelect(this.selectedCell);
    }
  }
}

window.ANMSGridHeatmap = ANMSGridHeatmap;
