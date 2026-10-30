/**
 * ChandraAlign - Cinematic 3D Space Background Environment (Three.js)
 * 
 * Features:
 * - Ultra-dense, brilliant field of 5,500+ stars highly visible across mid-brightness screens.
 * - Dynamic animated scintillation & twinkling stellar clusters.
 * - 10 Prominent Optical Guide Stars with 4-point cross diffraction spikes and lens glints.
 * - Active small & medium 3D irregular rocky meteoroids drifting and tumbling along the side flanks.
 * - Realistic Aerodynamic Falling Meteoroids on Click: Smooth tapered plasma tail, incandescent tumbling rock core, and glowing ion spark wake (Zero square artifacts).
 * - Periodic ambient shooting stars gliding through the side sectors.
 * - Distant, subdued 3D lunar limb in the bottom corner.
 * - Visually quiet central area behind the dashboard cards for maximum text contrast.
 * - High-performance THREE.Points & THREE.InstancedMesh.
 * - Responsive & accessible (prefers-reduced-motion).
 */

class CinematicSpaceEngine {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.canvas.id = canvasId;
      document.body.prepend(this.canvas);
    }

    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this.time = 0;

    // Bolide / Shooting Meteor System
    this.bolides = [];
    this.lastAmbientBolideTime = Date.now();
    this.nextAmbientInterval = 2800 + Math.random() * 2500;

    // Accessibility check: Reduced motion preference
    this.reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (typeof THREE !== 'undefined') {
      this.initThreeJS();
    } else {
      console.warn('Three.js library not loaded; falling back to 2D canvas');
      this.initCanvas2DFallback();
    }
  }

  /* ========================================================================
     THREE.JS ENGINE INITIALIZATION
     ======================================================================== */
  initThreeJS() {
    this.isWebGL = true;

    // 1. Deep Space Scene & Fog
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x06090F, 0.0003);

    this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 1, 4000);
    this.camera.position.set(0, 0, 720);

    // 2. High-Performance WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x06090F, 1.0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35; // Enhanced clarity for mid-brightness displays

    // 3. Procedural Textures
    this.textures = {
      lunarSurface: this.generateSubduedLunarTexture(512, 256),
      asteroidRock: this.generateAsteroidRockTexture(256, 256),
      starSprite: this.generateHighLuminanceStarSprite(),
      sparkleSprite: this.generateSparkleStarSprite(),
      meteorFlare: this.generateSmoothMeteorFlareTexture(),
      meteorTrail: this.generateTaperedMeteorTrailTexture(),
      diffractionSpike: this.generateOpticalSpikeTexture(),
      subtleDust: this.generateSubtleDustTexture()
    };

    // 4. Directional Lighting & Rim Fill
    this.setupLighting();

    // 5. Construct Scene Elements
    this.createUltraDenseTwinklingStarfield();
    this.createOpticalGuideStars();
    this.createSubtleDeepSpaceDust();
    this.createDistantLunarLimb();
    this.createSideFlankMeteoroidBelt();

    // 6. Bind Events & Animation Loop
    this.bindEvents();
    this.animate();
  }

  setupLighting() {
    // Ambient deep-space fill
    const ambientLight = new THREE.AmbientLight(0x182232, 0.65);
    this.scene.add(ambientLight);

    // Primary Grazing Sun Light (Top-Left)
    this.sunLight = new THREE.DirectionalLight(0xFFFFFF, 3.0);
    this.sunLight.position.set(-550, 380, 320);
    this.scene.add(this.sunLight);

    // Secondary Matrix Cyan Rim Light (Gives clear edge visibility to side asteroids)
    this.cyanRimLight = new THREE.DirectionalLight(0x66FCF1, 0.95);
    this.cyanRimLight.position.set(400, -250, 180);
    this.scene.add(this.cyanRimLight);
  }

  /* ------------------------------------------------------------------------
     1. High-Luminance Procedural Texture Helpers
     ------------------------------------------------------------------------ */
  generateHighLuminanceStarSprite() {
    // High-contrast, bright core for mid-brightness screen visibility
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
    grad.addColorStop(0, '#FFFFFF');
    grad.addColorStop(0.35, 'rgba(255, 255, 255, 0.98)');
    grad.addColorStop(0.65, 'rgba(102, 252, 241, 0.55)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
  }

  generateSparkleStarSprite() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const cx = 32, cy = 32;

    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 26);
    grad.addColorStop(0, '#FFFFFF');
    grad.addColorStop(0.35, 'rgba(255, 255, 255, 0.95)');
    grad.addColorStop(0.7, 'rgba(102, 252, 241, 0.45)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    // Brilliant 4-point cross glint
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(8, cy); ctx.lineTo(56, cy);
    ctx.moveTo(cx, 8); ctx.lineTo(cx, 56);
    ctx.stroke();

    return new THREE.CanvasTexture(canvas);
  }

  generateOpticalSpikeTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const cx = 64, cy = 64;

    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 32);
    grad.addColorStop(0, '#FFFFFF');
    grad.addColorStop(0.3, 'rgba(255, 255, 255, 0.98)');
    grad.addColorStop(0.65, 'rgba(102, 252, 241, 0.5)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);

    // Crisp diffraction spikes
    ctx.strokeStyle = '#66FCF1';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = '#66FCF1';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(4, cy); ctx.lineTo(124, cy);
    ctx.moveTo(cx, 4); ctx.lineTo(cx, 124);
    ctx.stroke();

    // Subtle 45-degree secondary spikes
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.beginPath();
    ctx.moveTo(22, 22); ctx.lineTo(106, 106);
    ctx.moveTo(106, 22); ctx.lineTo(22, 106);
    ctx.stroke();

    return new THREE.CanvasTexture(canvas);
  }

  generateSmoothMeteorFlareTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const cx = 64, cy = 64;

    const imgData = ctx.createImageData(128, 128);
    const data = imgData.data;
    for (let y = 0; y < 128; y++) {
      for (let x = 0; x < 128; x++) {
        const dx = (x - cx) / 52;
        const dy = (y - cy) / 52;
        const distSq = dx * dx + dy * dy;
        const alpha = Math.max(0, Math.exp(-distSq * 3.8));
        const idx = (y * 128 + x) * 4;
        data[idx] = 255;
        data[idx + 1] = 255;
        data[idx + 2] = 255;
        data[idx + 3] = Math.floor(alpha * 255);
      }
    }
    ctx.putImageData(imgData, 0, 0);
    return new THREE.CanvasTexture(canvas);
  }

  generateTaperedMeteorTrailTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const imgData = ctx.createImageData(256, 64);
    const data = imgData.data;
    for (let y = 0; y < 64; y++) {
      const latDist = Math.abs(y - 32) / 32;
      const latAlpha = Math.max(0, 1 - latDist * latDist);
      for (let x = 0; x < 256; x++) {
        const longRatio = x / 256;
        const longAlpha = Math.pow(1 - longRatio, 1.8);
        const combined = latAlpha * longAlpha;

        const idx = (y * 256 + x) * 4;
        data[idx] = Math.floor(255 * (1 - longRatio * 0.5));
        data[idx + 1] = Math.floor(255 * (1 - longRatio * 0.1));
        data[idx + 2] = 255;
        data[idx + 3] = Math.floor(combined * 255);
      }
    }
    ctx.putImageData(imgData, 0, 0);
    return new THREE.CanvasTexture(canvas);
  }

  generateSubtleDustTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(128, 128, 10, 128, 128, 128);
    grad.addColorStop(0, 'rgba(69, 162, 158, 0.14)');
    grad.addColorStop(0.45, 'rgba(20, 30, 48, 0.06)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(canvas);
  }

  generateSubduedLunarTexture(w = 512, h = 256) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    const baseGrad = ctx.createLinearGradient(0, 0, w, h);
    baseGrad.addColorStop(0, '#94A3B8');
    baseGrad.addColorStop(0.4, '#475569');
    baseGrad.addColorStop(0.8, '#1E293B');
    baseGrad.addColorStop(1, '#070A0F');
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, w, h);

    for (let i = 0; i < 50; i++) {
      const cx = Math.random() * w;
      const cy = Math.random() * h;
      const r = Math.random() * 24 + 3;

      const rimGrad = ctx.createLinearGradient(cx - r, cy - r, cx + r, cy + r);
      rimGrad.addColorStop(0, 'rgba(255, 255, 255, 0.65)');
      rimGrad.addColorStop(0.4, 'rgba(148, 163, 184, 0.4)');
      rimGrad.addColorStop(1, 'rgba(5, 8, 12, 0.85)');
      ctx.fillStyle = rimGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#05080E';
      ctx.beginPath();
      ctx.arc(cx + r * 0.15, cy + r * 0.15, r * 0.72, 0, Math.PI * 2);
      ctx.fill();
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }

  generateAsteroidRockTexture(w = 256, h = 256) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#475569';
    ctx.fillRect(0, 0, w, h);

    for (let i = 0; i < 60; i++) {
      ctx.strokeStyle = Math.random() > 0.5 ? 'rgba(255, 255, 255, 0.35)' : 'rgba(0, 0, 0, 0.5)';
      ctx.lineWidth = Math.random() * 2.5 + 1;
      ctx.beginPath();
      let px = Math.random() * w;
      let py = Math.random() * h;
      ctx.moveTo(px, py);
      px += (Math.random() - 0.5) * 60;
      py += (Math.random() - 0.5) * 60;
      ctx.lineTo(px, py);
      ctx.stroke();
    }

    for (let i = 0; i < 30; i++) {
      const px = Math.random() * w;
      const py = Math.random() * h;
      const pr = Math.random() * 7 + 2;
      ctx.fillStyle = '#0F172A';
      ctx.beginPath();
      ctx.arc(px, py, pr, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }

  /* ------------------------------------------------------------------------
     2. Ultra-Dense Twinkling GPU Starfield (5,500+ Stars)
     ------------------------------------------------------------------------ */
  createUltraDenseTwinklingStarfield() {
    // Layer 1: Dense Pinpoint Dust Stars (3,600 points) - Solid, bright visibility
    const count1 = 3600;
    const geom1 = new THREE.BufferGeometry();
    const pos1 = new Float32Array(count1 * 3);
    const col1 = new Float32Array(count1 * 3);

    const palette = [
      new THREE.Color('#FFFFFF'),
      new THREE.Color('#FFFFFF'),
      new THREE.Color('#F0F9FF'),
      new THREE.Color('#BAE6FD'),
      new THREE.Color('#66FCF1'),
      new THREE.Color('#A5F3FC'),
      new THREE.Color('#FEF08A')
    ];

    for (let i = 0; i < count1; i++) {
      pos1[i * 3] = (Math.random() - 0.5) * 3200;
      pos1[i * 3 + 1] = (Math.random() - 0.5) * 2400;
      pos1[i * 3 + 2] = (Math.random() - 0.5) * 1600 - 300;

      const c = palette[Math.floor(Math.random() * palette.length)];
      col1[i * 3] = c.r;
      col1[i * 3 + 1] = c.g;
      col1[i * 3 + 2] = c.b;
    }

    geom1.setAttribute('position', new THREE.BufferAttribute(pos1, 3));
    geom1.setAttribute('color', new THREE.BufferAttribute(col1, 3));

    const mat1 = new THREE.PointsMaterial({
      size: 3.2,
      vertexColors: true,
      map: this.textures.starSprite,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.starField1 = new THREE.Points(geom1, mat1);
    this.scene.add(this.starField1);

    // Layer 2: Sparkling Twinkling Star Clusters (1,900 points)
    const count2 = 1900;
    const geom2 = new THREE.BufferGeometry();
    const pos2 = new Float32Array(count2 * 3);
    const col2 = new Float32Array(count2 * 3);

    for (let i = 0; i < count2; i++) {
      pos2[i * 3] = (Math.random() - 0.5) * 2800;
      pos2[i * 3 + 1] = (Math.random() - 0.5) * 2000;
      pos2[i * 3 + 2] = (Math.random() - 0.5) * 1200 - 100;

      const c = palette[Math.floor(Math.random() * palette.length)];
      col2[i * 3] = c.r;
      col2[i * 3 + 1] = c.g;
      col2[i * 3 + 2] = c.b;
    }

    geom2.setAttribute('position', new THREE.BufferAttribute(pos2, 3));
    geom2.setAttribute('color', new THREE.BufferAttribute(col2, 3));

    this.twinkleMat = new THREE.PointsMaterial({
      size: 4.5,
      vertexColors: true,
      map: this.textures.sparkleSprite,
      transparent: true,
      opacity: 0.92,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.starField2 = new THREE.Points(geom2, this.twinkleMat);
    this.scene.add(this.starField2);
  }

  createOpticalGuideStars() {
    this.guideStars = [];
    const guideConfigs = [
      { x: -420, y: 260, z: -260, size: 44, speed: 0.02 },
      { x: 440, y: 240, z: -220, size: 48, speed: 0.018 },
      { x: 480, y: -180, z: -180, size: 40, speed: 0.022 },
      { x: -460, y: -160, z: -200, size: 38, speed: 0.025 },
      { x: 400, y: 30, z: -160, size: 34, speed: 0.02 },
      { x: -500, y: 70, z: -240, size: 38, speed: 0.016 },
      { x: 180, y: 320, z: -300, size: 36, speed: 0.024 },
      { x: -200, y: 340, z: -320, size: 40, speed: 0.019 },
      { x: 520, y: -80, z: -250, size: 36, speed: 0.021 },
      { x: -520, y: -40, z: -270, size: 34, speed: 0.023 }
    ];

    guideConfigs.forEach(cfg => {
      const mat = new THREE.SpriteMaterial({
        map: this.textures.diffractionSpike,
        color: 0xFFFFFF,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(cfg.x, cfg.y, cfg.z);
      sprite.scale.set(cfg.size, cfg.size, 1);
      this.scene.add(sprite);

      this.guideStars.push({
        sprite: sprite,
        baseSize: cfg.size,
        phase: Math.random() * Math.PI * 2,
        speed: cfg.speed
      });
    });
  }

  /* ------------------------------------------------------------------------
     3. Subtle Deep Space Dust Filaments
     ------------------------------------------------------------------------ */
  createSubtleDeepSpaceDust() {
    this.dustGroup = new THREE.Group();
    const count = 10;

    for (let i = 0; i < count; i++) {
      const mat = new THREE.SpriteMaterial({
        map: this.textures.subtleDust,
        color: i % 2 === 0 ? 0x45A29E : 0x1F2833,
        transparent: true,
        opacity: 0.12,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const sprite = new THREE.Sprite(mat);

      const angle = (i / count) * Math.PI * 2;
      const radius = 680 + (Math.random() - 0.5) * 160;
      sprite.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * (radius * 0.65),
        -550
      );
      const s = Math.random() * 450 + 350;
      sprite.scale.set(s, s, 1);
      this.dustGroup.add(sprite);
    }

    this.scene.add(this.dustGroup);
  }

  /* ------------------------------------------------------------------------
     4. Distant 3D Lunar Limb (<10% Viewport, Far Bottom-Left)
     ------------------------------------------------------------------------ */
  createDistantLunarLimb() {
    const moonGroup = new THREE.Group();
    const radius = 120;

    const geometry = new THREE.SphereGeometry(radius, 40, 40);
    const material = new THREE.MeshStandardMaterial({
      map: this.textures.lunarSurface,
      roughness: 0.94,
      metalness: 0.06,
      bumpMap: this.textures.lunarSurface,
      bumpScale: 2.2
    });

    this.lunarMesh = new THREE.Mesh(geometry, material);
    moonGroup.add(this.lunarMesh);

    const limbGeom = new THREE.SphereGeometry(radius * 1.018, 24, 24);
    const limbMat = new THREE.MeshBasicMaterial({
      color: 0x66FCF1,
      transparent: true,
      opacity: 0.1,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide
    });
    moonGroup.add(new THREE.Mesh(limbGeom, limbMat));

    moonGroup.position.set(-370, -280, -420);
    this.scene.add(moonGroup);
    this.lunarGroup = moonGroup;
  }

  /* ------------------------------------------------------------------------
     5. Active Small & Medium Side-Flank 3D Meteoroids (THREE.InstancedMesh)
     ------------------------------------------------------------------------ */
  createAsteroidBaseGeometry() {
    const geom = new THREE.IcosahedronGeometry(1.0, 2);
    const pos = geom.attributes.position;
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      const n1 = Math.sin(v.x * 2.4) * Math.cos(v.y * 2.4) * 0.32;
      const n2 = Math.cos(v.z * 3.2) * 0.18;
      v.multiplyScalar(1.0 + n1 + n2);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geom.computeVertexNormals();
    return geom;
  }

  createSideFlankMeteoroidBelt() {
    this.asteroidCount = 64;
    const baseGeom = this.createAsteroidBaseGeometry();
    const material = new THREE.MeshStandardMaterial({
      color: 0xCBD5E1,
      roughness: 0.88,
      metalness: 0.12,
      map: this.textures.asteroidRock,
      bumpMap: this.textures.asteroidRock,
      bumpScale: 3.2,
      flatShading: true
    });

    this.asteroidInstancedMesh = new THREE.InstancedMesh(baseGeom, material, this.asteroidCount);
    this.asteroidInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    this.asteroidData = [];
    const dummy = new THREE.Object3D();

    for (let i = 0; i < this.asteroidCount; i++) {
      let scale;
      if (i < 12) scale = Math.random() * 14 + 20; // Medium
      else if (i < 48) scale = Math.random() * 8 + 10; // Small
      else scale = Math.random() * 4 + 5; // Shards

      let posX, posY;
      const isLeft = i % 2 === 0;

      if (isLeft) {
        posX = -340 - Math.random() * 320;
        posY = (Math.random() - 0.45) * 850;
      } else {
        posX = 340 + Math.random() * 320;
        posY = (Math.random() - 0.55) * 850;
      }

      const posZ = (Math.random() - 0.5) * 380 - 30;
      const angle = Math.PI * (1.14 + (Math.random() - 0.5) * 0.22);
      const speed = (Math.random() * 0.45 + 0.25) * (this.reducedMotion ? 0.05 : 1.0);

      const data = {
        isLeft: isLeft,
        x: posX,
        y: posY,
        z: posZ,
        scale: scale,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed * 0.65,
        vz: (Math.random() - 0.5) * 0.05,
        rotX: Math.random() * Math.PI * 2,
        rotY: Math.random() * Math.PI * 2,
        rotZ: Math.random() * Math.PI * 2,
        rotSpeedX: (Math.random() - 0.5) * (this.reducedMotion ? 0.001 : 0.008),
        rotSpeedY: (Math.random() - 0.5) * (this.reducedMotion ? 0.001 : 0.01),
        rotSpeedZ: (Math.random() - 0.5) * (this.reducedMotion ? 0.001 : 0.006)
      };

      this.asteroidData.push(data);

      dummy.position.set(data.x, data.y, data.z);
      dummy.rotation.set(data.rotX, data.rotY, data.rotZ);
      dummy.scale.set(data.scale, data.scale, data.scale);
      dummy.updateMatrix();
      this.asteroidInstancedMesh.setMatrixAt(i, dummy.matrix);
    }

    this.asteroidInstancedMesh.instanceMatrix.needsUpdate = true;
    this.scene.add(this.asteroidInstancedMesh);
  }

  /* ------------------------------------------------------------------------
     6. Photorealistic Aerodynamic Falling Meteoroids & Bolides
     ------------------------------------------------------------------------ */
  spawnBolide(targetClientX = null, targetClientY = null) {
    let startX, startY;
    if (targetClientX !== null && targetClientY !== null) {
      const normX = (targetClientX / this.width) * 2 - 1;
      const normY = -(targetClientY / this.height) * 2 + 1;
      startX = normX * 420 + (Math.random() * 100 + 60);
      startY = normY * 300 + (Math.random() * 120 + 80);
    } else {
      const side = Math.random() > 0.5 ? 1 : -1;
      startX = side * (300 + Math.random() * 280);
      startY = 360 + Math.random() * 140;
    }

    const startZ = (Math.random() - 0.5) * 160 - 20;
    const angle = Math.PI * (1.15 + (Math.random() - 0.5) * 0.14);
    const speed = Math.random() * 18 + 22;
    const trailLength = Math.random() * 180 + 120;

    const meteorGroup = new THREE.Group();
    meteorGroup.position.set(startX, startY, startZ);
    meteorGroup.rotation.z = angle + Math.PI;

    // A. 3D Tumbling Rock Core
    const rockGeom = new THREE.IcosahedronGeometry(2.4, 1);
    const rockMat = new THREE.MeshStandardMaterial({
      color: 0xE2E8F0,
      roughness: 0.8,
      metalness: 0.2,
      flatShading: true
    });
    const rockMesh = new THREE.Mesh(rockGeom, rockMat);
    meteorGroup.add(rockMesh);

    // B. Incandescent Smooth Gaussian Plasma Head
    const flareMat = new THREE.SpriteMaterial({
      map: this.textures.meteorFlare,
      color: 0xFFFFFF,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending
    });
    const flareSprite = new THREE.Sprite(flareMat);
    flareSprite.scale.set(22, 22, 1);
    meteorGroup.add(flareSprite);

    // C. Aerodynamic Tapered Smooth Plasma Trail Mesh
    const trailGeom = new THREE.PlaneGeometry(trailLength, 7, 1, 1);
    trailGeom.translate(trailLength / 2, 0, 0);
    const trailMat = new THREE.MeshBasicMaterial({
      map: this.textures.meteorTrail,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const trailMesh = new THREE.Mesh(trailGeom, trailMat);
    meteorGroup.add(trailMesh);

    this.scene.add(meteorGroup);

    this.bolides.push({
      group: meteorGroup,
      rock: rockMesh,
      flare: flareSprite,
      trail: trailMesh,
      x: startX,
      y: startY,
      z: startZ,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: 0,
      maxLife: Math.floor(Math.random() * 28 + 26)
    });
  }

  /* ------------------------------------------------------------------------
     7. Handlers, Responsive Resizing & Parallax Loop
     ------------------------------------------------------------------------ */
  bindEvents() {
    window.addEventListener('resize', () => {
      this.width = window.innerWidth;
      this.height = window.innerHeight;
      this.camera.aspect = this.width / this.height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.width, this.height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    });

    window.addEventListener('click', (e) => {
      this.spawnBolide(e.clientX, e.clientY);
    });

    if (!this.reducedMotion) {
      window.addEventListener('mousemove', (e) => {
        this.mouse.targetX = (e.clientX / this.width) * 2 - 1;
        this.mouse.targetY = -(e.clientY / this.height) * 2 + 1;
      });

      window.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0) {
          this.mouse.targetX = (e.touches[0].clientX / this.width) * 2 - 1;
          this.mouse.targetY = -(e.touches[0].clientY / this.height) * 2 + 1;
        }
      }, { passive: true });
    }

    if (window.matchMedia) {
      const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
      mediaQuery.addEventListener('change', () => {
        this.reducedMotion = mediaQuery.matches;
      });
    }
  }

  animate() {
    this.time += 0.01;

    // 1. Subtle, Damped Parallax
    if (!this.reducedMotion) {
      this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.035;
      this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.035;

      this.camera.position.x = this.mouse.x * 25;
      this.camera.position.y = this.mouse.y * 18;
      this.camera.lookAt(0, 0, 0);
    }

    // 2. Slow Lunar Rotation
    if (this.lunarMesh) {
      this.lunarMesh.rotation.y += this.reducedMotion ? 0.00003 : 0.00012;
    }

    // 3. Dynamic Star Scintillation & Guide Star Twinkle
    if (this.twinkleMat) {
      this.twinkleMat.opacity = 0.8 + 0.2 * Math.sin(this.time * 3.5);
    }

    this.guideStars.forEach(g => {
      g.phase += g.speed;
      const s = g.baseSize * (0.88 + 0.22 * Math.sin(g.phase));
      g.sprite.scale.set(s, s, 1);
    });

    // 4. Update Side-Flank 3D Asteroids
    if (this.asteroidInstancedMesh && this.asteroidData) {
      const dummy = new THREE.Object3D();

      for (let i = 0; i < this.asteroidCount; i++) {
        const a = this.asteroidData[i];
        a.x += a.vx;
        a.y += a.vy;
        a.z += a.vz;

        a.rotX += a.rotSpeedX;
        a.rotY += a.rotSpeedY;
        a.rotZ += a.rotSpeedZ;

        if (a.x < -720 || a.y < -550) {
          if (a.isLeft) {
            a.x = -320 - Math.random() * 300;
            a.y = 400 + Math.random() * 220;
          } else {
            a.x = 340 + Math.random() * 300;
            a.y = 400 + Math.random() * 220;
          }
          a.z = (Math.random() - 0.5) * 380 - 30;
        }

        dummy.position.set(a.x, a.y, a.z);
        dummy.rotation.set(a.rotX, a.rotY, a.rotZ);
        dummy.scale.set(a.scale, a.scale, a.scale);
        dummy.updateMatrix();

        this.asteroidInstancedMesh.setMatrixAt(i, dummy.matrix);
      }

      this.asteroidInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    // 5. Update Falling Meteoroids / Bolides
    const now = Date.now();
    if (now - this.lastAmbientBolideTime > this.nextAmbientInterval) {
      this.spawnBolide();
      this.lastAmbientBolideTime = now;
      this.nextAmbientInterval = 2500 + Math.random() * 3000;
    }

    for (let i = this.bolides.length - 1; i >= 0; i--) {
      const b = this.bolides[i];
      b.x += b.vx;
      b.y += b.vy;
      b.life++;

      const lifeRatio = b.life / b.maxLife;
      const alpha = Math.max(0, 1.0 - lifeRatio);

      b.group.position.set(b.x, b.y, b.z);
      b.flare.material.opacity = alpha;
      b.trail.material.opacity = alpha * 0.9;

      b.rock.rotation.x += 0.12;
      b.rock.rotation.y += 0.15;

      if (b.life >= b.maxLife || b.x < -800 || b.y < -600) {
        this.scene.remove(b.group);
        b.rock.geometry.dispose();
        b.rock.material.dispose();
        b.flare.material.dispose();
        b.trail.geometry.dispose();
        b.trail.material.dispose();
        this.bolides.splice(i, 1);
      }
    }

    // 6. Slow Dust Swirl
    if (this.dustGroup) {
      this.dustGroup.rotation.z = Math.sin(this.time * 0.04) * 0.015;
    }

    // Render WebGL Frame
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.animate());
  }

  /* ========================================================================
     CANVAS 2D FALLBACK (When WebGL is unavailable)
     ======================================================================== */
  initCanvas2DFallback() {
    this.ctx = this.canvas.getContext('2d');
    const loop = () => {
      this.ctx.fillStyle = '#06090F';
      this.ctx.fillRect(0, 0, this.width, this.height);
      requestAnimationFrame(loop);
    };
    loop();
  }
}

// Auto-initialize when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  window.cinematicSpace = new CinematicSpaceEngine('space-starfield-bg');
});
