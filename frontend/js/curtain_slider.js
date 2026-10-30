/**
 * ChandraAlign - High-Fidelity Split-Screen Curtain Slider
 * Real-time sub-pixel comparison between Chandrayaan-2 optical swath and LROC reference map.
 */

class CurtainSliderInspector {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.options = Object.assign({
      initialPosition: 0.5,
      enableDiffMode: false,
      onPositionChange: null
    }, options);

    this.position = this.options.initialPosition;
    this.isDragging = false;
    this.zoom = 1.0;
    this.diffMode = false;
    this.isRegisteredMode = true; // true = aligned (0.26px), false = raw unaligned (14.28px offset)
    this.nudgeOffset = { x: 0, y: 0 };

    this.cvsCh2 = null;
    this.cvsLroc = null;
    this.cvsDiff = null;

    this.initDOM();
    this.bindEvents();
  }

  initDOM() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="curtain-slider-container verification-split-container" id="curtain-viewport">
        <!-- Layer 1: LROC Reference (Underlay) -->
        <div class="curtain-layer" id="layer-lroc">
          <canvas id="cvs-curtain-lroc"></canvas>
        </div>

        <!-- Layer 2: CH-2 Optical / Registered (Clipped Overlay) -->
        <div class="curtain-layer" id="layer-ch2" style="clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);">
          <canvas id="cvs-curtain-ch2"></canvas>
        </div>

        <!-- Layer 3: Sub-pixel Difference Overlay -->
        <div class="curtain-layer" id="layer-diff" style="display: none; mix-blend-mode: color-dodge; opacity: 0.85;">
          <canvas id="cvs-curtain-diff"></canvas>
        </div>

        <!-- Draggable Neon Divider Bar -->
        <div class="curtain-divider-bar" id="curtain-divider" style="left: 50%;">
          <div class="curtain-handle-knob">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="8 16 4 12 8 8"></polyline>
              <polyline points="16 8 20 12 16 16"></polyline>
            </svg>
          </div>
        </div>

        <!-- Overlay HUD Badges -->
        <div class="canvas-hud-overlay" style="top: 12px; left: 12px;">
          <span style="color: var(--accent-cyan); font-weight: bold;">◀ CH-2 OHRC (0.25m/px REGISTERED)</span>
        </div>
        <div class="canvas-hud-overlay" style="top: 12px; right: 12px; left: auto;">
          <span style="color: var(--accent-amber); font-weight: bold;">LROC NAC (0.50m/px REFERENCE) ▶</span>
        </div>
        <div class="canvas-coords-overlay" id="curtain-stats-hud">
          SPLIT: 50.0% | ZOOM: 1.0x | NUDGE: (0.00px, 0.00px)
        </div>
      </div>
    `;

    this.viewport = document.getElementById('curtain-viewport');
    this.divider = document.getElementById('curtain-divider');
    this.layerCh2 = document.getElementById('layer-ch2');
    this.layerLroc = document.getElementById('layer-lroc');
    this.layerDiff = document.getElementById('layer-diff');

    this.cvsCh2 = document.getElementById('cvs-curtain-ch2');
    this.cvsLroc = document.getElementById('cvs-curtain-lroc');
    this.cvsDiff = document.getElementById('cvs-curtain-diff');
  }

  bindEvents() {
    if (!this.viewport || !this.divider) return;

    const onMove = (e) => {
      if (!this.isDragging) return;
      const rect = this.viewport.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      let ratio = (clientX - rect.left) / rect.width;
      ratio = Math.max(0.00, Math.min(1.00, ratio));
      this.setPosition(ratio);
    };

    const onEnd = () => {
      this.isDragging = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onEnd);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onEnd);
    };

    this.divider.addEventListener('mousedown', (e) => {
      e.preventDefault();
      this.isDragging = true;
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onEnd);
    });

    this.divider.addEventListener('touchstart', (e) => {
      this.isDragging = true;
      window.addEventListener('touchmove', onMove);
      window.addEventListener('touchend', onEnd);
    }, { passive: true });

    // Allow dragging by clicking or dragging anywhere across the full responsive container
    this.viewport.addEventListener('mousedown', (e) => {
      if (e.target.closest('.canvas-hud-overlay, .canvas-coords-overlay')) return;
      this.isDragging = true;
      const rect = this.viewport.getBoundingClientRect();
      let ratio = (e.clientX - rect.left) / rect.width;
      ratio = Math.max(0.00, Math.min(1.00, ratio));
      this.setPosition(ratio);
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onEnd);
    });

    this.viewport.addEventListener('touchstart', (e) => {
      if (e.target.closest('.canvas-hud-overlay, .canvas-coords-overlay')) return;
      this.isDragging = true;
      const rect = this.viewport.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : 0;
      let ratio = (clientX - rect.left) / rect.width;
      ratio = Math.max(0.00, Math.min(1.00, ratio));
      this.setPosition(ratio);
      window.addEventListener('touchmove', onMove);
      window.addEventListener('touchend', onEnd);
    }, { passive: true });

    // Responsive container auto-resize observer
    if (window.ResizeObserver && this.viewport) {
      this.resizeObserver = new ResizeObserver(() => {
        this.setupCanvases();
        this.redraw();
      });
      this.resizeObserver.observe(this.viewport);
    } else {
      window.addEventListener('resize', () => {
        this.setupCanvases();
        this.redraw();
      });
    }

    this.viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      this.zoom = Math.max(0.5, Math.min(4.0, this.zoom * zoomFactor));
      this.updateHUD();
      this.redraw();
    });
  }

  setPosition(ratio) {
    this.position = ratio;
    const pct = (ratio * 100).toFixed(2);
    if (this.divider) this.divider.style.left = `${pct}%`;
    if (this.layerCh2) this.layerCh2.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
    this.updateHUD();
  }

  nudge(dx, dy) {
    this.nudgeOffset.x += dx;
    this.nudgeOffset.y += dy;
    this.updateHUD();
    this.redraw();
  }

  resetNudge() {
    this.nudgeOffset = { x: 0, y: 0 };
    this.zoom = 1.0;
    this.updateHUD();
    this.redraw();
  }

  toggleDiffMode(active) {
    this.diffMode = active;
    if (this.layerDiff) {
      this.layerDiff.style.display = active ? 'block' : 'none';
    }
  }

  updateHUD() {
    const stats = document.getElementById('curtain-stats-hud');
    if (stats) {
      if (this.realImages && this.realImages.is_registered === false) {
        stats.innerHTML = `<span style="color:var(--accent-rose); font-weight:bold;">⚠️ REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED</span> | SPLIT: ${(this.position * 100).toFixed(1)}% | ZOOM: ${this.zoom.toFixed(1)}x`;
      } else {
        stats.textContent = `SPLIT: ${(this.position * 100).toFixed(1)}% | ZOOM: ${this.zoom.toFixed(1)}x | NUDGE: (${this.nudgeOffset.x.toFixed(2)}px, ${this.nudgeOffset.y.toFixed(2)}px)`;
      }
    }
  }

  loadData(presetData, realImages = null) {
    this.presetData = presetData;
    if (realImages) this.realImages = realImages;
    this.setupCanvases();
    this.updateHUD();
    this.redraw();
  }

  setRealImages(realImages) {
    this.realImages = realImages;
    this.updateHUD();
    this.redraw();
  }

  setupCanvases() {
    if (!this.viewport) return;
    const rect = this.viewport.getBoundingClientRect();
    const width = Math.round(rect.width || this.viewport.clientWidth || 900);
    const height = Math.round(rect.height || this.viewport.clientHeight || (window.innerHeight * 0.8) || 560);

    [this.cvsCh2, this.cvsLroc, this.cvsDiff].forEach(c => {
      if (c && (c.width !== width || c.height !== height)) {
        c.width = width;
        c.height = height;
      }
    });
  }

  setRegistrationMode(isRegistered) {
    this.isRegisteredMode = isRegistered;
    this.updateHUD();
    this.redraw();
  }

  drawRealImage(ctx, w, h, img, zoom, offset = { x: 0, y: 0 }) {
    if (!img) return;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#060a10';
    ctx.fillRect(0, 0, w, h);
    ctx.translate(w / 2, h / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-w / 2 + (offset.x || 0), -h / 2 + (offset.y || 0));

    const natW = img.naturalWidth || img.width || w;
    const natH = img.naturalHeight || img.height || h;
    const scale = Math.min(w / natW, h / natH);
    const drawW = natW * scale;
    const drawH = natH * scale;
    ctx.drawImage(img, (w - drawW) / 2, (h - drawH) / 2, drawW, drawH);
    ctx.restore();
  }

  redraw() {
    if (!this.cvsCh2 || !this.cvsLroc) return;
    const ctxCh2 = this.cvsCh2.getContext('2d');
    const ctxLroc = this.cvsLroc.getContext('2d');
    const ctxDiff = this.cvsDiff?.getContext('2d');

    const w = this.cvsCh2.width;
    const h = this.cvsCh2.height;

    // LROC Reference Layer
    if (this.realImages && this.realImages.ref) {
      this.drawRealImage(ctxLroc, w, h, this.realImages.ref, this.zoom, { x: 0, y: 0 });
    } else {
      this.drawLunarShadedRelief(ctxLroc, w, h, {
        isLroc: true,
        zoom: this.zoom,
        offset: { x: 0, y: 0 },
        craters: this.presetData?.craters || []
      });
    }

    // CH-2 Layer (apply unaligned sensor shift if not registered)
    const rawOffset = this.isRegisteredMode ? { x: 0, y: 0 } : { x: 14.28, y: -8.91 };
    const effectiveOffset = {
      x: this.nudgeOffset.x + rawOffset.x,
      y: this.nudgeOffset.y + rawOffset.y
    };

    if (this.realImages && (this.realImages.warped || this.realImages.src)) {
      const ch2Img = (this.isRegisteredMode && this.realImages.warped) ? this.realImages.warped : (this.realImages.src || this.realImages.warped);
      this.drawRealImage(ctxCh2, w, h, ch2Img, this.zoom, effectiveOffset);
    } else {
      this.drawLunarShadedRelief(ctxCh2, w, h, {
        isLroc: false,
        zoom: this.zoom,
        offset: effectiveOffset,
        rotationRad: this.isRegisteredMode ? 0 : 0.000733,
        craters: this.presetData?.craters || []
      });
    }

    // Difference map
    if (ctxDiff) {
      if (this.realImages && this.realImages.diff) {
        this.drawRealImage(ctxDiff, w, h, this.realImages.diff, this.zoom, { x: 0, y: 0 });
      } else {
        this.drawDifferenceLayer(ctxDiff, w, h);
      }
    }
  }

  drawLunarShadedRelief(ctx, w, h, options) {
    ctx.save();
    ctx.clearRect(0, 0, w, h);

    ctx.translate(w / 2, h / 2);
    ctx.scale(options.zoom, options.zoom);
    if (options.rotationRad) {
      ctx.rotate(options.rotationRad);
    }
    ctx.translate(-w / 2 + (options.offset.x || 0), -h / 2 + (options.offset.y || 0));

    // 1. High-Contrast Lunar Regolith Basemap with Anorthosite Highlands & Basaltic Maria
    const bgGrad = ctx.createLinearGradient(0, 0, w, h);
    if (options.isLroc) {
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
    const flowCount = options.isLroc ? 12 : 16;
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
    const seed = options.isLroc ? 53 : 101;
    for (let i = 0; i < 1400; i++) {
      const rx = (((Math.sin(i * 131 + seed) * 10000) % 1 + 1) % 1) * w;
      const ry = (((Math.cos(i * 313 + seed) * 10000) % 1 + 1) % 1) * h;
      const rad = ((i % 6) + 1) * 0.65;
      const lum = ((i % 8) + 1) / 8;

      ctx.fillStyle = options.isLroc
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
    const craters = options.craters || [];
    craters.forEach((crater) => {
      const center = options.isLroc ? crater.lroc_center : crater.ch2_center;
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
      ejectaGrad.addColorStop(0, options.isLroc ? 'rgba(197, 198, 199, 0.26)' : 'rgba(255, 255, 255, 0.32)');
      ejectaGrad.addColorStop(0.5, options.isLroc ? 'rgba(69, 162, 158, 0.08)' : 'rgba(102, 252, 241, 0.1)');
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

    // 5. Epipolar Photogrammetric Grid Guidelines
    ctx.strokeStyle = options.isLroc ? 'rgba(69, 162, 158, 0.08)' : 'rgba(102, 252, 241, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 60) {
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    ctx.restore();
  }

  drawDifferenceLayer(ctx, w, h) {
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0, 242, 254, 0.18)';
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }
}

window.CurtainSliderInspector = CurtainSliderInspector;
