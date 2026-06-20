# Author: Antigravity AI
# Reads 'model_weights.json' and embeds it as a JS constant inside 'app.js'

import json
import os

print("--- EMBEDDING NEURAL NETWORK WEIGHTS INTO APP.JS ---")

weights_file = "model_weights.json"
if not os.path.exists(weights_file):
    print(f"Error: {weights_file} not found! Please run the training script first.")
    exit(1)

with open(weights_file, "r") as f:
    weights = json.load(f)

# Write the core javascript logic and inject the weights
js_content = f"""/*
 * app.js
 * Edge AI Inference Engine & Neon Runner Game
 * Author: Antigravity AI
 */

// Embedded trained neural network weights (MLP Classifier)
const MODEL_WEIGHTS = {json.dumps(weights, indent=2)};

const CLASS_NAMES = ['Closed', 'Three', 'Open', 'Zero'];

// Audio Synthesis Utility (Web Audio API)
class RetroAudioSynth {{
  constructor() {{
    this.ctx = null;
    this.enabled = true;
  }}

  init() {{
    if (!this.ctx) {{
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }}
  }}

  toggle(val) {{
    this.enabled = val;
    if (this.enabled) {{
      this.init();
    }}
  }}

  playJump() {{
    if (!this.enabled) return;
    this.init();
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(150, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(800, this.ctx.currentTime + 0.15);
    
    gain.gain.setValueAtTime(0.15, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.18);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.18);
  }}

  playSlide() {{
    if (!this.enabled) return;
    this.init();
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(120, this.ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(40, this.ctx.currentTime + 0.25);
    
    gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.28);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.28);
  }}

  playCrash() {{
    if (!this.enabled) return;
    this.init();
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(100, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(20, this.ctx.currentTime + 0.5);
    
    gain.gain.setValueAtTime(0.3, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.5);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.5);
  }}

  playSelect() {{
    if (!this.enabled) return;
    this.init();
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(523.25, this.ctx.currentTime); // C5
    osc.frequency.setValueAtTime(659.25, this.ctx.currentTime + 0.08); // E5
    
    gain.gain.setValueAtTime(0.1, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.2);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.2);
  }}
}}

const synth = new RetroAudioSynth();

// --- 1. NEURAL NETWORK INFERENCE ENGINE ---

// Performs matrix multiplication for feedforward pass of the MLP neural network
function predictMLP(x) {{
  // Layer 1: Dense (32 units, ReLU activation)
  // z1 = x * W1 + b1
  const hiddenSize = 32;
  const numClasses = 4;
  
  let z1 = new Array(hiddenSize).fill(0);
  for (let j = 0; j < hiddenSize; j++) {{
    let sum = 0;
    for (let i = 0; i < 42; i++) {{
      sum += x[i] * MODEL_WEIGHTS.W1[i][j];
    }}
    z1[j] = Math.max(0, sum + MODEL_WEIGHTS.b1[j]); // ReLU
  }}

  // Layer 2: Dense (4 units, Softmax activation)
  // z2 = z1 * W2 + b2
  let z2 = new Array(numClasses).fill(0);
  let maxLogit = -Infinity;
  for (let j = 0; j < numClasses; j++) {{
    let sum = 0;
    for (let i = 0; i < hiddenSize; i++) {{
      sum += z1[i] * MODEL_WEIGHTS.W2[i][j];
    }}
    z2[j] = sum + MODEL_WEIGHTS.b2[j];
    if (z2[j] > maxLogit) {{
      maxLogit = z2[j];
    }}
  }}

  // Compute Softmax probabilities
  let expSum = 0;
  let probs = new Array(numClasses).fill(0);
  for (let j = 0; j < numClasses; j++) {{
    probs[j] = Math.exp(z2[j] - maxLogit);
    expSum += probs[j];
  }}
  
  for (let j = 0; j < numClasses; j++) {{
    probs[j] /= expSum;
  }}
  
  return probs;
}}

// Translation and scale-invariant coordinates normalization
function preprocessLandmarks(landmarks) {{
  const wrist = landmarks[0];
  let centered = [];
  let maxDist = 0;
  
  // Center all landmarks relative to the wrist (landmark 0)
  for (let i = 0; i < 21; i++) {{
    const lm = landmarks[i];
    const dx = lm.x - wrist.x;
    const dy = lm.y - wrist.y;
    centered.push({{ x: dx, y: dy }});
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > maxDist) {{
      maxDist = dist;
    }}
  }}
  
  // Scale normalize relative to the maximum distance
  let flattened = [];
  for (let i = 0; i < 21; i++) {{
    if (maxDist > 0) {{
      flattened.push(centered[i].x / maxDist);
      flattened.push(centered[i].y / maxDist);
    }} else {{
      flattened.push(0);
      flattened.push(0);
    }}
  }}
  
  return {{ flattened, wrist, scale: maxDist }};
}}

// --- 2. DYNAMIC NEON ARCADE SIMULATOR GAME ---

class NeonRunnerGame {{
  constructor(canvasId) {{
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.score = 0;
    this.highScore = parseInt(localStorage.getItem('gesture_runner_high') || '0');
    this.isPlaying = false;
    this.obstacles = [];
    this.animationId = null;
    this.speed = 4;
    this.frameCount = 0;
    
    // Player settings
    this.player = {{
      x: 100,
      y: 0, // dynamic ground coordinate
      width: 26,
      height: 38,
      originalHeight: 38,
      slidingHeight: 18,
      vy: 0,
      gravity: 0.7,
      jumpStrength: -11.5,
      isJumping: false,
      isSliding: false,
      vx: 0,
      speedX: 5,
      trail: []
    }};
    
    this.groundY = this.canvas.height - 50;
    this.player.y = this.groundY - this.player.height;
    
    this.gridOffset = 0;
  }}

  start() {{
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.score = 0;
    this.speed = 4.5;
    this.obstacles = [];
    this.player.x = 100;
    this.player.vx = 0;
    this.player.vy = 0;
    this.player.isJumping = false;
    this.player.isSliding = false;
    this.player.height = this.player.originalHeight;
    this.player.y = this.groundY - this.player.height;
    this.player.trail = [];
    this.frameCount = 0;
    
    synth.playSelect();
    
    // Hide startup instructions on game canvas
    this.tick();
  }}

  stop() {{
    this.isPlaying = false;
    if (this.animationId) {{
      cancelAnimationFrame(this.animationId);
    }}
    synth.playCrash();
    
    // Check High Score
    if (this.score > this.highScore) {{
      this.highScore = this.score;
      localStorage.setItem('gesture_runner_high', this.highScore);
    }}
    
    this.drawGameOver();
  }}

  reset() {{
    this.score = 0;
    this.isPlaying = false;
    this.obstacles = [];
    this.player.x = 100;
    this.player.vx = 0;
    this.player.vy = 0;
    this.player.isJumping = false;
    this.player.isSliding = false;
    this.player.height = this.player.originalHeight;
    this.player.y = this.groundY - this.player.height;
    this.player.trail = [];
    
    synth.playSelect();
    
    if (this.animationId) {{
      cancelAnimationFrame(this.animationId);
    }}
    
    this.drawMenu();
    this.updateStats();
  }}

  triggerAction(action) {{
    if (!this.isPlaying) return;
    
    if (action === 'up') {{
      if (!this.player.isJumping && !this.player.isSliding) {{
        this.player.vy = this.player.jumpStrength;
        this.player.isJumping = true;
        synth.playJump();
      }}
    }} else if (action === 'down') {{
      if (!this.player.isJumping && !this.player.isSliding) {{
        this.player.isSliding = true;
        this.player.height = this.player.slidingHeight;
        this.player.y += (this.player.originalHeight - this.player.slidingHeight);
        synth.playSlide();
      }}
    }} else if (action === 'left') {{
      this.player.vx = -this.player.speedX;
    }} else if (action === 'right') {{
      this.player.vx = this.player.speedX;
    }}
  }}

  releaseAction(action) {{
    if (!this.isPlaying) return;
    
    if (action === 'down') {{
      if (this.player.isSliding) {{
        this.player.isSliding = false;
        this.player.y -= (this.player.originalHeight - this.player.slidingHeight);
        this.player.height = this.player.originalHeight;
      }}
    }} else if (action === 'left' || action === 'right') {{
      this.player.vx = 0;
    }}
  }}

  update() {{
    this.frameCount++;
    
    // Increment score
    if (this.frameCount % 5 === 0) {{
      this.score++;
      this.updateStats();
    }}
    
    // Scale speed slowly
    if (this.frameCount % 400 === 0) {{
      this.speed += 0.5;
    }}
    
    // Horizontal Movement Bounds
    this.player.x += this.player.vx;
    if (this.player.x < 20) this.player.x = 20;
    if (this.player.x > this.canvas.width - 50) this.player.x = this.canvas.width - 50;
    
    // Gravity and Vertical Jump Physics
    this.player.y += this.player.vy;
    this.player.vy += this.player.gravity;
    
    const currentHeight = this.player.height;
    if (this.player.y > this.groundY - currentHeight) {{
      this.player.y = this.groundY - currentHeight;
      this.player.vy = 0;
      this.player.isJumping = false;
    }}
    
    // Handle player tail/trail particles
    this.player.trail.push({{ x: this.player.x + this.player.width/2, y: this.player.y + this.player.height/2 }});
    if (this.player.trail.length > 12) {{
      this.player.trail.shift();
    }}
    
    // Scroll ground grid lines
    this.gridOffset = (this.gridOffset + this.speed) % 40;
    
    // Spawn Obstacles
    if (this.frameCount % 100 === 0 || (this.frameCount % 137 === 0 && Math.random() > 0.6)) {{
      const type = Math.random() > 0.45 ? 'ground' : 'floating';
      if (type === 'ground') {{
        this.obstacles.push({{
          x: this.canvas.width + 10,
          y: this.groundY - 30,
          width: 20,
          height: 30,
          type: 'ground'
        }});
      }} else {{
        this.obstacles.push({{
          x: this.canvas.width + 10,
          y: this.groundY - 65,
          width: 30,
          height: 18,
          type: 'floating'
        }});
      }}
    }}
    
    // Update Obstacles
    for (let i = this.obstacles.length - 1; i >= 0; i--) {{
      const obs = this.obstacles[i];
      obs.x -= this.speed;
      
      // Collision check
      if (
        this.player.x < obs.x + obs.width &&
        this.player.x + this.player.width > obs.x &&
        this.player.y < obs.y + obs.height &&
        this.player.y + this.player.height > obs.y
      ) {{
        this.stop();
        return;
      }}
      
      // Clean up out of bounds obstacles
      if (obs.x < -40) {{
        this.obstacles.splice(i, 1);
      }}
    }}
  }}

  draw() {{
    this.ctx.fillStyle = '#04020a';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // 1. Draw glowing cyberpunk sky background grid lines
    this.ctx.strokeStyle = 'rgba(224, 0, 255, 0.12)';
    this.ctx.lineWidth = 1;
    for (let x = 0; x < this.canvas.width; x += 40) {{
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.groundY);
      this.ctx.stroke();
    }}
    
    // 2. Draw ground road perspective and moving horizontal lines
    this.ctx.fillStyle = '#0c071a';
    this.ctx.fillRect(0, this.groundY, this.canvas.width, this.canvas.height - this.groundY);
    
    this.ctx.strokeStyle = '#e000ff';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(0, this.groundY);
    this.ctx.lineTo(this.canvas.width, this.groundY);
    this.ctx.stroke();
    
    // Draw ground speed grid horizontal lines
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
    this.ctx.lineWidth = 1.5;
    for (let y = this.groundY; y < this.canvas.height; y += 10) {{
      const ratio = (y - this.groundY) / (this.canvas.height - this.groundY);
      const dy = this.groundY + ratio * (this.canvas.height - this.groundY) - (this.gridOffset * ratio) % 15;
      if (dy >= this.groundY) {{
        this.ctx.beginPath();
        this.ctx.moveTo(0, dy);
        this.ctx.lineTo(this.canvas.width, dy);
        this.ctx.stroke();
      }}
    }}
    
    // Draw perspective lane split lines
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';
    const centerX = this.canvas.width / 2;
    for (let offset = -400; offset <= 400; offset += 100) {{
      this.ctx.beginPath();
      this.ctx.moveTo(centerX + offset * 0.1, this.groundY);
      this.ctx.lineTo(centerX + offset, this.canvas.height);
      this.ctx.stroke();
    }}
    
    // 3. Draw player trail particles
    if (this.player.trail.length > 1) {{
      this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
      this.ctx.lineWidth = 4;
      this.ctx.lineCap = 'round';
      this.ctx.beginPath();
      this.ctx.moveTo(this.player.trail[0].x, this.player.trail[0].y);
      for (let i = 1; i < this.player.trail.length; i++) {{
        this.ctx.lineTo(this.player.trail[i].x, this.player.trail[i].y);
      }}
      this.ctx.stroke();
    }}
    
    // 4. Draw Player character (Glowing Cyberpunk Triangle/Hexagon)
    this.ctx.save();
    this.ctx.shadowBlur = 15;
    this.ctx.shadowColor = '#00f0ff';
    this.ctx.fillStyle = '#00f0ff';
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = 2;
    
    this.ctx.beginPath();
    if (this.player.isSliding) {{
      // Sliding: Drawn as a sleek low rectangle
      this.ctx.roundRect(this.player.x, this.player.y, this.player.width, this.player.height, 4);
      this.ctx.fill();
      this.ctx.stroke();
    }} else {{
      // Standing/Jumping: Sleek cyber triangle
      this.ctx.moveTo(this.player.x + this.player.width / 2, this.player.y);
      this.ctx.lineTo(this.player.x, this.player.y + this.player.height);
      this.ctx.lineTo(this.player.x + this.player.width, this.player.y + this.player.height);
      this.ctx.closePath();
      this.ctx.fill();
      this.ctx.stroke();
    }}
    this.ctx.restore();
    
    // 5. Draw Obstacles (Glowing Pink Barriers/Spikes)
    this.obstacles.forEach(obs => {{
      this.ctx.save();
      this.ctx.shadowBlur = 12;
      this.ctx.shadowColor = '#ff007f';
      this.ctx.fillStyle = '#ff007f';
      this.ctx.strokeStyle = '#ffffff';
      this.ctx.lineWidth = 1.5;
      
      this.ctx.beginPath();
      if (obs.type === 'ground') {{
        // Ground Spikes: Triangle shapes
        this.ctx.moveTo(obs.x + obs.width / 2, obs.y);
        this.ctx.lineTo(obs.x, obs.y + obs.height);
        this.ctx.lineTo(obs.x + obs.width, obs.y + obs.height);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
      }} else {{
        // Floating Bars: Rounded rectangle shields
        this.ctx.roundRect(obs.x, obs.y, obs.width, obs.height, 4);
        this.ctx.fill();
        this.ctx.stroke();
      }}
      this.ctx.restore();
    }});
  }}

  drawMenu() {{
    this.ctx.fillStyle = '#04020a';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Background Grid
    this.ctx.strokeStyle = 'rgba(224, 0, 255, 0.08)';
    this.ctx.lineWidth = 1;
    for (let x = 0; x < this.canvas.width; x += 40) {{
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.canvas.height);
      this.ctx.stroke();
    }}
    
    // Welcome Text
    this.ctx.fillStyle = '#00f0ff';
    this.ctx.font = '800 24px "Outfit"';
    this.ctx.textAlign = 'center';
    this.ctx.shadowBlur = 10;
    this.ctx.shadowColor = '#00f0ff';
    this.ctx.fillText("NEON RUNNER ENGINE", this.canvas.width / 2, this.canvas.height / 2 - 40);
    
    this.ctx.fillStyle = '#ff007f';
    this.ctx.font = '800 13px "Share Tech Mono"';
    this.ctx.shadowColor = '#ff007f';
    this.ctx.fillText("CONTROLLED BY NEURAL HAND GESTURES", this.canvas.width / 2, this.canvas.height / 2 - 10);
    
    this.ctx.fillStyle = '#8e8a9f';
    this.ctx.font = '12px "Outfit"';
    this.ctx.shadowBlur = 0;
    this.ctx.fillText("1. GRANT WEBCAM PERMISSIONS", this.canvas.width / 2, this.canvas.height / 2 + 30);
    this.ctx.fillText("2. CLICK 'START SIMULATOR' TO BEGIN PLAYING", this.canvas.width / 2, this.canvas.height / 2 + 50);
  }}

  drawGameOver() {{
    this.ctx.fillStyle = 'rgba(5, 4, 9, 0.8)';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.ctx.fillStyle = '#ff007f';
    this.ctx.font = '800 32px "Outfit"';
    this.ctx.textAlign = 'center';
    this.ctx.shadowBlur = 15;
    this.ctx.shadowColor = '#ff007f';
    this.ctx.fillText("CRASH DETECTED", this.canvas.width / 2, this.canvas.height / 2 - 30);
    
    this.ctx.fillStyle = '#f0edf6';
    this.ctx.font = '16px "Share Tech Mono"';
    this.ctx.shadowBlur = 0;
    this.ctx.fillText(`SCORE ACCUMULATED: ${{this.score}}m`, this.canvas.width / 2, this.canvas.height / 2 + 10);
    
    this.ctx.fillStyle = '#8e8a9f';
    this.ctx.font = '12px "Outfit"';
    this.ctx.fillText("PERFORM GESTURE OR PRESS START TO RETRY", this.canvas.width / 2, this.canvas.height / 2 + 50);
  }}

  updateStats() {{
    document.getElementById('game-score').textContent = `${{this.score}}m`;
    document.getElementById('game-highscore').textContent = `${{this.highScore}}m`;
  }}

  tick() {{
    if (!this.isPlaying) return;
    this.update();
    this.draw();
    this.animationId = requestAnimationFrame(() => this.tick());
  }}
}}

const game = new NeonRunnerGame('arcade-canvas');
game.drawMenu();

// --- 3. UI CONTROLLER & EVENT LISTENERS ---

let activeCommand = null;

// Audio Controls
const audioBtn = document.getElementById('audio-toggle');
audioBtn.addEventListener('click', () => {{
  synth.init();
  if (synth.enabled) {{
    synth.toggle(false);
    audioBtn.textContent = '🔇 AUDIO: OFF';
    audioBtn.style.color = '#8e8a9f';
  }} else {{
    synth.toggle(true);
    audioBtn.textContent = '🔊 AUDIO: ON';
    audioBtn.style.color = '#fffb00';
  }}
}});

// Game Controls
const startBtn = document.getElementById('game-start-btn');
const resetBtn = document.getElementById('game-reset-btn');

startBtn.addEventListener('click', () => {{
  synth.init();
  game.start();
}});

resetBtn.addEventListener('click', () => {{
  synth.init();
  game.reset();
}});

// Custom Map select configurations
const gestureMappings = {{
  '0': 'up',    // Closed -> Jump
  '1': 'right', // Three -> Right
  '2': 'left',  // Open -> Left
  '3': 'down'   // Zero -> Slide
}};

document.querySelectorAll('.gesture-mapping-select').forEach(select => {{
  select.addEventListener('change', (e) => {{
    const classId = e.target.getAttribute('data-class');
    gestureMappings[classId] = e.target.value;
  }});
}});

// Update Direction UI highlights
function updateDirectionUI(activeAction) {{
  const dirs = ['up', 'down', 'left', 'right'];
  dirs.forEach(d => {{
    const el = document.getElementById(`dir-${{d}}`);
    if (d === activeAction) {{
      el.classList.add('active');
    }} else {{
      el.classList.remove('active');
    }}
  }});
}}

// --- 4. WEBCAM & MEDIAPE HANDS SYSTEM ---

const videoElement = document.getElementById('webcam-video');
const canvasElement = document.getElementById('canvas-overlay');
const canvasCtx = canvasElement.getContext('2d');
const sysStatusDot = document.getElementById('sys-status-dot');
const sysStatusText = document.getElementById('sys-status-text');
const trackingText = document.getElementById('tracking-text');

// Resize drawing canvas overlay to match video dimensions
function resizeOverlayCanvas() {{
  canvasElement.width = videoElement.clientWidth || 640;
  canvasElement.height = videoElement.clientHeight || 480;
}}

videoElement.addEventListener('loadedmetadata', resizeOverlayCanvas);
window.addEventListener('resize', resizeOverlayCanvas);

// Draws detailed hand skeleton on HTML5 canvas
function drawHandSkeleton(landmarks) {{
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  if (!landmarks) return;
  
  const width = canvasElement.width;
  const height = canvasElement.height;
  
  // Custom futuristic drawing specs
  const jointColor = '#00f0ff';
  const connectionColor = 'rgba(255, 0, 127, 0.7)';
  
  // Connection line definitions (standard 21 landmark pairs)
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [0, 17]                               // Palm
  ];
  
  // Draw Connections
  canvasCtx.strokeStyle = connectionColor;
  canvasCtx.lineWidth = 3;
  connections.forEach(conn => {{
    const pt1 = landmarks[conn[0]];
    const pt2 = landmarks[conn[1]];
    
    canvasCtx.beginPath();
    canvasCtx.moveTo(pt1.x * width, pt1.y * height);
    canvasCtx.lineTo(pt2.x * width, pt2.y * height);
    canvasCtx.stroke();
  }});
  
  // Draw Joints
  landmarks.forEach(lm => {{
    canvasCtx.fillStyle = jointColor;
    canvasCtx.shadowBlur = 8;
    canvasCtx.shadowColor = jointColor;
    canvasCtx.beginPath();
    canvasCtx.arc(lm.x * width, lm.y * height, 5, 0, 2 * Math.PI);
    canvasCtx.fill();
  }});
  canvasCtx.shadowBlur = 0; // reset
}}

// Setup prediction and loop callback for MediaPipe Hands
function onHandResults(results) {{
  resizeOverlayCanvas();
  
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {{
    sysStatusDot.className = 'status-dot active';
    sysStatusText.textContent = 'HAND DETECTED';
    trackingText.textContent = 'HAND DETECTED';
    
    const landmarks = results.multiHandLandmarks[0];
    
    // Draw joints and bones skeleton
    drawHandSkeleton(landmarks);
    
    // Preprocess landmarks (Centering and scaling)
    const {{ flattened, wrist, scale }} = preprocessLandmarks(landmarks);
    
    // Update calibration visual outputs
    document.getElementById('calib-wrist').textContent = `X: ${{wrist.x.toFixed(2)}}, Y: ${{wrist.y.toFixed(2)}}`;
    document.getElementById('calib-scale').textContent = `${{scale.toFixed(3)}}`;
    
    // Run the MLP Neural Network inference engine
    const probabilities = predictMLP(flattened);
    
    // Find the highest confidence prediction class index
    let maxIdx = 0;
    let maxVal = -1;
    for (let i = 0; i < probabilities.length; i++) {{
      const probPercent = Math.round(probabilities[i] * 100);
      document.getElementById(`bar-${{i}}`).style.width = `${{probPercent}}%`;
      document.getElementById(`val-${{i}}`).textContent = `${{probPercent}}%`;
      
      if (probabilities[i] > maxVal) {{
        maxVal = probabilities[i];
        maxIdx = i;
      }}
    }}
    
    // Set UI highlights
    for (let i = 0; i < 4; i++) {{
      const el = document.getElementById(`gesture-${{i}}`);
      if (i === maxIdx && maxVal > 0.6) {{
        el.classList.add('active');
      }} else {{
        el.classList.remove('active');
      }}
    }}
    
    // High-confidence filter (> 60% confidence threshold)
    if (maxVal > 0.60) {{
      const gestureClass = maxIdx.toString();
      const mappedAction = gestureMappings[gestureClass];
      
      if (mappedAction !== 'none') {{
        if (activeCommand !== mappedAction) {{
          // Release previous command if exist
          if (activeCommand) {{
            game.releaseAction(activeCommand);
          }}
          // Activate new action
          activeCommand = mappedAction;
          game.triggerAction(activeCommand);
          updateDirectionUI(activeCommand);
        }}
      }} else {{
        if (activeCommand) {{
          game.releaseAction(activeCommand);
          activeCommand = null;
          updateDirectionUI(null);
        }}
      }}
    }} else {{
      if (activeCommand) {{
        game.releaseAction(activeCommand);
        activeCommand = null;
        updateDirectionUI(null);
      }}
    }}
    
  }} else {{
    // No hand detected in frame
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    sysStatusDot.className = 'status-dot processing';
    sysStatusText.textContent = 'VISION SCANNING';
    trackingText.textContent = 'No hand detected';
    
    document.getElementById('calib-wrist').textContent = '-';
    document.getElementById('calib-scale').textContent = '-';
    
    // Reset all bars to 0%
    for (let i = 0; i < 4; i++) {{
      document.getElementById(`bar-${{i}}`).style.width = '0%';
      document.getElementById(`val-${{i}}`).textContent = '0%';
      document.getElementById(`gesture-${{i}}`).classList.remove('active');
    }}
    
    // Release active gesture commands
    if (activeCommand) {{
      game.releaseAction(activeCommand);
      activeCommand = null;
      updateDirectionUI(null);
    }}
  }}
}}

// Initialize MediaPipe Hands solution
const hands = new Hands({{
  locateFile: (file) => {{
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${{file}}`;
  }}
}});

hands.setOptions({{
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
}});

hands.onResults(onHandResults);

// Setup camera capturing loops using standard mediaDevices APIs
const camera = new Camera(videoElement, {{
  onFrame: async () => {{
    await hands.send({{ image: videoElement }});
  }},
  width: 640,
  height: 480
}});

camera.start()
  .then(() => {{
    sysStatusDot.className = 'status-dot active';
    sysStatusText.textContent = 'SYSTEM ONLINE';
    trackingText.textContent = 'Camera initialized';
  }})
  .catch(err => {{
    console.error("Camera Init Error:", err);
    sysStatusDot.className = 'status-dot';
    sysStatusText.textContent = 'CAMERA ERROR';
    trackingText.textContent = 'Camera initialization failed!';
  }});
"""

with open("app.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print("Generated finished app.js with embedded weights successfully!")
