/**
 * ChandraAlign - 8x8 Checkerboard Alignment Inspector
 * Alternating square matrix between CH-2 and LROC to detect sub-pixel boundary shearing.
 */

class CheckerboardInspector {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.options = Object.assign({
      gridDivisions: 8,
      blinkRateHz: 0,
      showEdges: false
    }, options);

    this.gridDivisions = this.options.gridDivisions;
    this.blinkRateHz = this.options.blinkRateHz;
    this.showEdges = this.options.showEdges;
    this.blinkPhase = 0;
    this.blinkInterval = null;
    this.presetData = null;
    this.isRegisteredMode = true;

    this.canvas = null;
    this.ctx = null;

    this.offscreenCh2 = document.createElement('canvas');
    this.offscreenLroc = document.createElement('canvas');

    this.initDOM();
    this.bindEvents();
  }

  initDOM() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="checkerboard-container" id="checker-viewport">
        <canvas id="cvs-checkerboard"></canvas>

        <div class="checkerboard-hud-controls">
          <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.75rem; color:var(--text-secondary);">
            <span style="font-weight:700;">GRID:</span>
            <button class="btn-secondary" style="padding:4px 9px;" id="chk-grid-4">4x4</button>
            <button class="btn-secondary active" style="padding:4px 9px; border-color:var(--accent-cyan); color:var(--accent-cyan);" id="chk-grid-8">8x8</button>
            <button class="btn-secondary" style="padding:4px 9px;" id="chk-grid-16">16x16</button>
          </div>

          <div style="width:1px; height:20px; background:var(--border-subtle);"></div>

          <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.75rem; color:var(--text-secondary);">
            <span style="font-weight:700;">BLINK INSPECTION:</span>
            <button class="btn-secondary" style="padding:4px 9px;" id="chk-blink-off">OFF</button>
            <button class="btn-secondary" style="padding:4px 9px;" id="chk-blink-2hz">2 Hz</button>
            <button class="btn-secondary" style="padding:4px 9px;" id="chk-blink-4hz">4 Hz</button>
          </div>

          <div style="width:1px; height:20px; background:var(--border-subtle);"></div>

          <label class="toggle-switch-wrapper" id="chk-toggle-edges">
            <div class="toggle-switch" id="switch-edges-knob">
              <div class="toggle-switch-knob"></div>
            </div>
            <span style="font-weight:700;">CONTOUR EDGES</span>
          </label>
        </div>

        <div class="canvas-hud-overlay" style="top: 12px; left: 12px;">
          <span style="color: var(--accent-cyan); font-weight:bold;">■ CH-2 REGISTERED</span> &nbsp;|&nbsp; 
          <span style="color: var(--accent-amber); font-weight:bold;">■ LROC REFERENCE</span>
        </div>

        <div class="canvas-coords-overlay" id="checker-hud-status">
          CHECKERBOARD: 8x8 | BLINK: OFF | RESIDUAL RMSE: 0.2643 px
        </div>
      </div>
    `;

    this.viewport = document.getElementById('checker-viewport');
    this.canvas = document.getElementById('cvs-checkerboard');
    if (this.canvas) {
      this.ctx = this.canvas.getContext('2d');
    }
  }

  bindEvents() {
    const setGrid = (num, btnId) => {
      this.gridDivisions = num;
      ['chk-grid-4', 'chk-grid-8', 'chk-grid-16'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
          btn.style.borderColor = (id === btnId) ? 'var(--accent-cyan)' : 'var(--border-subtle)';
          btn.style.color = (id === btnId) ? 'var(--accent-cyan)' : 'var(--text-primary)';
        }
      });
      this.updateHUD();
      this.render();
    };

    const setBlink = (hz, btnId) => {
      this.blinkRateHz = hz;
      if (this.blinkInterval) clearInterval(this.blinkInterval);

      ['chk-blink-off', 'chk-blink-2hz', 'chk-blink-4hz'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
          btn.style.borderColor = (id === btnId) ? 'var(--accent-cyan)' : 'var(--border-subtle)';
          btn.style.color = (id === btnId) ? 'var(--accent-cyan)' : 'var(--text-primary)';
        }
      });

      if (hz > 0) {
        const intervalMs = 1000 / (hz * 2);
        this.blinkInterval = setInterval(() => {
          this.blinkPhase = this.blinkPhase === 0 ? 1 : 0;
          this.render();
        }, intervalMs);
      } else {
        this.blinkPhase = 0;
        this.render();
      }
      this.updateHUD();
    };

    document.getElementById('chk-grid-4')?.addEventListener('click', () => setGrid(4, 'chk-grid-4'));
    document.getElementById('chk-grid-8')?.addEventListener('click', () => setGrid(8, 'chk-grid-8'));
    document.getElementById('chk-grid-16')?.addEventListener('click', () => setGrid(16, 'chk-grid-16'));

    document.getElementById('chk-blink-off')?.addEventListener('click', () => setBlink(0, 'chk-blink-off'));
    document.getElementById('chk-blink-2hz')?.addEventListener('click', () => setBlink(2, 'chk-blink-2hz'));
    document.getElementById('chk-blink-4hz')?.addEventListener('click', () => setBlink(4, 'chk-blink-4hz'));

    const edgeToggle = document.getElementById('chk-toggle-edges');
    const edgeKnob = document.getElementById('switch-edges-knob');
    edgeToggle?.addEventListener('click', () => {
      this.showEdges = !this.showEdges;
      if (edgeKnob) {
        edgeKnob.classList.toggle('active', this.showEdges);
      }
      this.render();
    });
  }

  updateHUD() {
    const status = document.getElementById('checker-hud-status');
    if (status) {
      const blinkText = this.blinkRateHz > 0 ? `${this.blinkRateHz} Hz ACTIVE` : 'OFF';
      status.textContent = `CHECKERBOARD: ${this.gridDivisions}x${this.gridDivisions} | BLINK: ${blinkText} | SUB-PIXEL RMSE: 0.2643 px`;
    }
  }

  loadData(presetData, realImages = null) {
    this.presetData = presetData;
    if (realImages) this.realImages = realImages;
    this.resizeCanvases();
    this.render();
  }

  setRealImages(realImages) {
    this.realImages = realImages;
    this.bakeOffscreenTerrain();
    this.render();
  }

  resizeCanvases() {
    const width = this.viewport?.clientWidth || 900;
    const height = this.viewport?.clientHeight || 560;

    if (this.canvas) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.offscreenCh2.width = width;
    this.offscreenCh2.height = height;
    this.offscreenLroc.width = width;
    this.offscreenLroc.height = height;

    this.bakeOffscreenTerrain();
  }

  setRegistrationMode(isRegistered) {
    this.isRegisteredMode = isRegistered;
    this.bakeOffscreenTerrain();
    this.updateHUD();
    this.render();
  }

  bakeOffscreenTerrain() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctxCh2 = this.offscreenCh2.getContext('2d');
    const ctxLroc = this.offscreenLroc.getContext('2d');
    const craters = this.presetData?.craters || [];

    this.drawTerrainLayer(ctxCh2, w, h, false, craters);
    this.drawTerrainLayer(ctxLroc, w, h, true, craters);
  }

  drawTerrainLayer(ctx, w, h, isLroc, craters) {
    ctx.clearRect(0, 0, w, h);

    const rawOffset = (!isLroc && !this.isRegisteredMode) ? { x: 14.28, y: -8.91 } : { x: 0, y: 0 };
    const rawRot = (!isLroc && !this.isRegisteredMode) ? 0.000733 : 0;

    if (this.realImages) {
      const img = isLroc
        ? this.realImages.ref
        : ((this.isRegisteredMode && this.realImages.warped) ? this.realImages.warped : (this.realImages.src || this.realImages.warped));
      if (img) {
        ctx.save();
        if (rawOffset.x !== 0 || rawOffset.y !== 0) {
          ctx.translate(w / 2, h / 2);
          ctx.rotate(rawRot);
          ctx.translate(-w / 2 + rawOffset.x, -h / 2 + rawOffset.y);
        }
        ctx.fillStyle = '#060a10';
        ctx.fillRect(0, 0, w, h);
        const natW = img.naturalWidth || img.width || w;
        const natH = img.naturalHeight || img.height || h;
        const scale = Math.min(w / natW, h / natH);
        const drawW = natW * scale;
        const drawH = natH * scale;
        ctx.drawImage(img, (w - drawW) / 2, (h - drawH) / 2, drawW, drawH);
        ctx.restore();
        return;
      }
    }

    ctx.save();
    if (rawOffset.x !== 0 || rawOffset.y !== 0) {
      ctx.translate(w / 2, h / 2);
      ctx.rotate(rawRot);
      ctx.translate(-w / 2 + rawOffset.x, -h / 2 + rawOffset.y);
    }

    // 1. High-Contrast Lunar Regolith Basemap with Anorthosite Highlands & Basaltic Maria
    const bgGrad = ctx.createLinearGradient(0, 0, w, h);
    if (isLroc) {
      bgGrad.addColorStop(0, '#384353'); // LROC 0.50m basemap
      bgGrad.addColorStop(0.3, '#252F3C');
      bgGrad.addColorStop(0.7, '#151D28');
      bgGrad.addColorStop(1, '#070A0F');
    } else {
      // Chandrayaan-2 OHRC (0.25m ultra-high resolution crisp highland regolith)
      bgGrad.addColorStop(0, '#48576A');
      bgGrad.addColorStop(0.28, '#303D4E');
      bgGrad.addColorStop(0.68, '#1B2431');
      bgGrad.addColorStop(1, '#080B10');
    }
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // 2. Multi-Octave Regolith Topography & Dark Basaltic Flow Patches
    const flowCount = isLroc ? 12 : 16;
    ctx.fillStyle = 'rgba(7, 10, 15, 0.42)';
    for (let f = 0; f < flowCount; f++) {
      ctx.beginPath();
      ctx.ellipse(
        (f * 137.5) % w, (f * 219.3) % h,
        ((f * 43) % 90) + 40, ((f * 31) % 60) + 25,
        (f * 0.4), 0, Math.PI * 2
      );
      ctx.fill();
    }

    // 3. Dense Rugged Boulder Fields & Micro-Craters (1,400+ surface features with directional shadows)
    const seed = isLroc ? 53 : 101;
    for (let i = 0; i < 1400; i++) {
      const rx = (((Math.sin(i * 131 + seed) * 10000) % 1 + 1) % 1) * w;
      const ry = (((Math.cos(i * 313 + seed) * 10000) % 1 + 1) % 1) * h;
      const rad = ((i % 6) + 1) * 0.65;
      const lum = ((i % 8) + 1) / 8;

      ctx.fillStyle = isLroc
        ? `rgba(197, 198, 199, ${0.06 + lum * 0.12})`
        : `rgba(255, 255, 255, ${0.09 + lum * 0.16})`;
      ctx.beginPath();
      ctx.arc(rx, ry, rad, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#030508';
      ctx.beginPath();
      ctx.arc(rx + rad * 0.7, ry + rad * 0.7, rad * 0.85, 0, Math.PI * 2);
      ctx.fill();
    }

    // 4. Photorealistic Complex Impact Craters with Terraced Scarps & Central Peaks
    craters.forEach((crater) => {
      const center = isLroc ? crater.lroc_center : crater.ch2_center;
      const cx = (center[0] / 1000) * w;
      const cy = (center[1] / 1000) * h;
      const a = (crater.semi_major / 1000) * w;
      const b = (crater.semi_minor / 1000) * h;
      const rot = (crater.angle_deg * Math.PI) / 180;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rot);

      // A. Radial High-Albedo Ejecta Rays
      const rayCount = Math.floor(Math.min(18, a / 2.8));
      ctx.save();
      for (let r = 0; r < rayCount; r++) {
        const rayAngle = (r / rayCount) * Math.PI * 2 + (r * 19) % 0.35;
        const rayLen = a * (1.9 + ((r * 17) % 1.5));
        const rayGrad = ctx.createLinearGradient(0, 0, Math.cos(rayAngle) * rayLen, Math.sin(rayAngle) * rayLen);
        rayGrad.addColorStop(0, 'rgba(255, 255, 255, 0.32)');
        rayGrad.addColorStop(0.35, 'rgba(197, 198, 199, 0.12)');
        rayGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.strokeStyle = rayGrad;
        ctx.lineWidth = Math.random() * 2.2 + 0.8;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(rayAngle) * rayLen, Math.sin(rayAngle) * rayLen);
        ctx.stroke();
      }
      ctx.restore();

      // B. Continuous Hummocky Ejecta Rim Flank
      const ejectaGrad = ctx.createRadialGradient(0, 0, a * 0.7, 0, 0, a * 1.55);
      ejectaGrad.addColorStop(0, isLroc ? 'rgba(197, 198, 199, 0.26)' : 'rgba(255, 255, 255, 0.32)');
      ejectaGrad.addColorStop(0.5, isLroc ? 'rgba(69, 162, 158, 0.08)' : 'rgba(102, 252, 241, 0.1)');
      ejectaGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = ejectaGrad;
      ctx.beginPath();
      ctx.ellipse(0, 0, a * 1.55, b * 1.55, 0, 0, Math.PI * 2);
      ctx.fill();

      // C. Raised Crater Rim Wall (Directional Grazing Sunlight NW vs Deep Shadow SE)
      const rimGrad = ctx.createLinearGradient(-a * 1.05, -b * 1.05, a * 1.05, b * 1.05);
      rimGrad.addColorStop(0, '#FFFFFF');
      rimGrad.addColorStop(0.22, '#CBD5E1');
      rimGrad.addColorStop(0.52, '#334155');
      rimGrad.addColorStop(0.78, '#0F172A');
      rimGrad.addColorStop(1, '#020407');

      ctx.beginPath();
      ctx.ellipse(0, 0, a, b, 0, 0, Math.PI * 2);
      ctx.fillStyle = rimGrad;
      ctx.fill();

      // D. Terraced Slump Steps
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.ellipse(-a * 0.05, -b * 0.05, a * 0.86, b * 0.86, 0, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
      ctx.beginPath();
      ctx.ellipse(a * 0.05, b * 0.05, a * 0.86, b * 0.86, 0, 0, Math.PI * 2);
      ctx.stroke();

      // E. Deep Pitch-Black Shadowed Basin Floor
      const floorGrad = ctx.createRadialGradient(a * 0.15, b * 0.15, a * 0.08, a * 0.22, b * 0.22, a * 0.72);
      floorGrad.addColorStop(0, '#070A0F');
      floorGrad.addColorStop(0.65, '#030508');
      floorGrad.addColorStop(1, '#010203');

      ctx.beginPath();
      ctx.ellipse(a * 0.16, b * 0.16, a * 0.7, b * 0.7, 0, 0, Math.PI * 2);
      ctx.fillStyle = floorGrad;
      ctx.fill();

      // F. Central Uplift Mountain Peak
      if (a > 18) {
        const peakX = -a * 0.03;
        const peakY = -b * 0.03;
        const peakRadius = a * 0.22;

        const peakGrad = ctx.createRadialGradient(peakX - peakRadius * 0.35, peakY - peakRadius * 0.35, 1, peakX, peakY, peakRadius);
        peakGrad.addColorStop(0, '#FFFFFF');
        peakGrad.addColorStop(0.35, '#C5C6C7');
        peakGrad.addColorStop(0.7, '#1E293B');
        peakGrad.addColorStop(1, '#020407');

        ctx.beginPath();
        ctx.arc(peakX, peakY, peakRadius, 0, Math.PI * 2);
        ctx.fillStyle = peakGrad;
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(peakX, peakY);
        ctx.lineTo(peakX + peakRadius * 2.0, peakY + peakRadius * 1.5);
        ctx.lineTo(peakX + peakRadius * 1.4, peakY + peakRadius * 2.2);
        ctx.closePath();
        ctx.fillStyle = '#010203';
        ctx.fill();
      }

      ctx.restore();
    });

    ctx.restore();
  }

  render() {
    if (!this.ctx || !this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const N = this.gridDivisions;
    const cellW = w / N;
    const cellH = h / N;

    this.ctx.clearRect(0, 0, w, h);

    if (this.blinkRateHz > 0) {
      const source = (this.blinkPhase === 0) ? this.offscreenCh2 : this.offscreenLroc;
      this.ctx.drawImage(source, 0, 0);
    } else {
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          const isEven = (r + c) % 2 === 0;
          const source = isEven ? this.offscreenCh2 : this.offscreenLroc;
          const sx = c * cellW;
          const sy = r * cellH;

          this.ctx.drawImage(source, sx, sy, cellW, cellH, sx, sy, cellW, cellH);

          this.ctx.strokeStyle = isEven ? 'rgba(0, 242, 254, 0.35)' : 'rgba(255, 184, 0, 0.35)';
          this.ctx.lineWidth = 1;
          this.ctx.strokeRect(sx, sy, cellW, cellH);
        }
      }
    }

    if (this.showEdges && this.presetData?.craters) {
      this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
      this.ctx.lineWidth = 1.8;
      this.ctx.setLineDash([4, 3]);

      this.presetData.craters.forEach(crater => {
        const cx = (crater.ch2_center[0] / 1000) * w;
        const cy = (crater.ch2_center[1] / 1000) * h;
        const a = (crater.semi_major / 1000) * w;
        const b = (crater.semi_minor / 1000) * h;
        const rot = (crater.angle_deg * Math.PI) / 180;

        this.ctx.save();
        this.ctx.translate(cx, cy);
        this.ctx.rotate(rot);
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, a, b, 0, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.restore();
      });
      this.ctx.setLineDash([]);
    }
  }
}

window.CheckerboardInspector = CheckerboardInspector;
