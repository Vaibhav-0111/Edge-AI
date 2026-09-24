/**
 * Edge AI PPE Safety & Hazard Triage Dashboard — Client Controller
 * Handles WebSocket connection, live telemetry sync, alert feed,
 * runtime PPE policies, and real-time Browser Webcam camera access & detection.
 */

class SafetyDashboard {
  constructor() {
    this.ws = null;
    this.reconnectInterval = 2000;
    this.activeFilter = "ALL";
    this.alerts = [];
    this.soundEnabled = true;
    this.audioCtx = null;
    this.currentSource = "demo"; // "demo", "client_webcam", "cam0"

    // Webcam stream state
    this.webcamStream = null;
    this.webcamInterval = null;
    this.isProcessingFrame = false;

    // DOM Elements
    this.statusDot = document.getElementById("statusDot");
    this.statusText = document.getElementById("statusText");
    this.fpsValue = document.getElementById("fpsValue");
    this.latencyValue = document.getElementById("latencyValue");
    this.latencySub = document.getElementById("latencySub");
    this.workersValue = document.getElementById("workersValue");
    this.violationsValue = document.getElementById("violationsValue");
    this.totalAlertsValue = document.getElementById("totalAlertsValue");
    this.alertsList = document.getElementById("alertsList");
    this.alertCountBadge = document.getElementById("alertCountBadge");
    this.videoFeed = document.getElementById("videoFeed");
    this.webcamVideo = document.getElementById("webcamVideo");
    this.webcamCanvas = document.getElementById("webcamCanvas");
    this.soundToggleBtn = document.getElementById("soundToggleBtn");
    this.camBadge = document.getElementById("camBadge");
    this.camSourceLabel = document.getElementById("camSourceLabel");

    this.initAudio();
    this.initWebSocket();
    this.bindEvents();
    this.loadInitialStatus();
  }

  initAudio() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContext();
    } catch (e) {
      console.warn("Web Audio API not supported", e);
    }
  }

  playAlertTone(severity) {
    if (!this.soundEnabled || !this.audioCtx) return;
    try {
      if (this.audioCtx.state === "suspended") {
        this.audioCtx.resume();
      }

      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      if (severity === "CRITICAL") {
        // High urgency alarm
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(880, this.audioCtx.currentTime);
        osc.frequency.setValueAtTime(440, this.audioCtx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.12, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.35);
        osc.start();
        osc.stop(this.audioCtx.currentTime + 0.35);
      } else if (severity === "HIGH") {
        // Single high-pitch chime
        osc.type = "sine";
        osc.frequency.setValueAtTime(587.33, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.25);
        osc.start();
        osc.stop(this.audioCtx.currentTime + 0.25);
      }
    } catch (e) {
      // Audio playback catch
    }
  }

  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "127.0.0.1:8000";
    const wsUrl = `${protocol}//${host}/ws`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("WebSocket connected successfully.");
      this.statusDot.classList.remove("disconnected");
      this.statusText.textContent = "EDGE PIPELINE ACTIVE";
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    this.ws.onclose = () => {
      console.warn("WebSocket disconnected. Retrying in 2s...");
      this.statusDot.classList.add("disconnected");
      this.statusText.textContent = "OFFLINE (RECONNECTING...)";
      setTimeout(() => this.initWebSocket(), this.reconnectInterval);
    };

    this.ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      this.ws.close();
    };
  }

  handleMessage(msg) {
    if (msg.type === "metrics") {
      this.updateMetrics(msg);
    } else if (msg.type === "alert") {
      this.addAlert(msg);
    } else if (msg.type === "history") {
      if (Array.isArray(msg.alerts)) {
        msg.alerts.forEach((alert) => this.addAlert(alert, false));
      }
    } else if (msg.type === "frame_sync") {
      if (msg.metrics) {
        this.updateMetrics(msg.metrics);
      }
    } else if (msg.type === "frame_result") {
      // Received direct frame response from live browser webcam inference
      this.isProcessingFrame = false;
      if (msg.annotated_image) {
        this.videoFeed.src = msg.annotated_image;
      }
      if (msg.metrics) {
        this.updateMetrics(msg.metrics);
      }
      if (Array.isArray(msg.alerts)) {
        msg.alerts.forEach((a) => this.addAlert(a));
      }
    }
  }

  updateMetrics(metrics) {
    if (metrics.fps !== undefined) {
      this.fpsValue.textContent = Number(metrics.fps).toFixed(1);
    }
    if (metrics.latency_ms !== undefined || metrics.total_latency_ms !== undefined) {
      const lat = metrics.latency_ms || metrics.total_latency_ms || 0;
      this.latencyValue.textContent = `${Number(lat).toFixed(1)} ms`;
      const inf = metrics.inference_ms ? `${Number(metrics.inference_ms).toFixed(1)}ms inf` : "ONNX CPU";
      this.latencySub.textContent = `${inf} | FP32`;
    }
    if (metrics.persons_detected !== undefined) {
      this.workersValue.textContent = metrics.persons_detected;
    }
    if (metrics.violations_active !== undefined) {
      this.violationsValue.textContent = metrics.violations_active;
    }
    if (metrics.total_alerts !== undefined) {
      this.totalAlertsValue.textContent = metrics.total_alerts;
    }
  }

  addAlert(alertData, playSound = true) {
    this.alerts.unshift(alertData);
    if (this.alerts.length > 80) {
      this.alerts.pop();
    }

    this.alertCountBadge.textContent = this.alerts.length;

    if (playSound && (alertData.severity === "CRITICAL" || alertData.severity === "HIGH")) {
      this.playAlertTone(alertData.severity);
    }

    this.renderAlerts();
  }

  renderAlerts() {
    const filtered = this.alerts.filter((a) => {
      if (this.activeFilter === "ALL") return true;
      return a.severity === this.activeFilter;
    });

    if (filtered.length === 0) {
      this.alertsList.innerHTML = `
        <div class="alert-empty-state">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <div>No confirmed ${this.activeFilter === "ALL" ? "" : this.activeFilter} alerts</div>
        </div>
      `;
      return;
    }

    this.alertsList.innerHTML = filtered
      .map((alert) => {
        const severityClass = `severity-${alert.severity}`;
        const badgeClass =
          alert.severity === "CRITICAL" ? "crit" :
          alert.severity === "HIGH" ? "high" :
          alert.severity === "MEDIUM" ? "med" : "low";

        const reasonTags = (alert.reasons || [])
          .map((r) => `<span class="pill-tag violation">${r.replace(/_/g, " ")}</span>`)
          .join("");

        const ppeTags = [];
        if (alert.helmet !== undefined && alert.helmet !== null) {
          ppeTags.push(`<span class="pill-tag">${alert.helmet ? "✓ Helmet" : "✗ No Helmet"}</span>`);
        }
        if (alert.vest !== undefined && alert.vest !== null) {
          ppeTags.push(`<span class="pill-tag">${alert.vest ? "✓ Vest" : "✗ No Vest"}</span>`);
        }
        if (alert.gloves !== undefined && alert.gloves !== null) {
          ppeTags.push(`<span class="pill-tag">${alert.gloves ? "✓ Gloves" : "✗ No Gloves"}</span>`);
        }

        const personId = alert.person_id ? `Worker #${alert.person_id}` : "Worker";

        return `
          <div class="alert-item ${severityClass}">
            <div class="alert-top">
              <span class="alert-badge ${badgeClass}">${alert.severity} (Score: ${alert.risk_score || 0})</span>
              <span class="alert-timestamp">${alert.timestamp || ""}</span>
            </div>
            <div class="alert-body">
              <strong>${personId}</strong> — ${alert.severity === "CRITICAL" ? "Immediate hazard override" : "Safety violation confirmed"}
            </div>
            <div class="alert-tags">
              ${reasonTags}
              ${ppeTags.join("")}
            </div>
          </div>
        `;
      })
      .join("");
  }

  bindEvents() {
    // Sound Mute Toggle
    if (this.soundToggleBtn) {
      this.soundToggleBtn.addEventListener("click", () => {
        this.soundEnabled = !this.soundEnabled;
        this.soundToggleBtn.innerHTML = this.soundEnabled
          ? `<span>🔊</span> Sound On`
          : `<span>🔇</span> Sound Off`;
      });
    }

    // Filter Tabs
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach((tab) => {
      tab.addEventListener("click", (e) => {
        tabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        this.activeFilter = tab.getAttribute("data-filter") || "ALL";
        this.renderAlerts();
      });
    });

    // PPE Policy Toggle Buttons
    const ppeBtns = document.querySelectorAll(".ppe-toggle-btn");
    ppeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.classList.toggle("active");
        this.syncPPEConfig();
      });
    });

    // Camera Source Selector
    const sourceBtns = document.querySelectorAll(".source-btn");
    sourceBtns.forEach((btn) => {
      btn.addEventListener("click", async () => {
        sourceBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const selectedSource = btn.getAttribute("data-source");
        await this.switchSource(selectedSource);
      });
    });

    // Video Feed Stream Error Recovery
    if (this.videoFeed) {
      this.videoFeed.onerror = () => {
        if (this.currentSource !== "client_webcam") {
          console.warn("Video stream stalled, reconnecting in 2s...");
          setTimeout(() => {
            this.videoFeed.src = `/api/video_feed?t=${Date.now()}`;
          }, 2000);
        }
      };
    }
  }

  async switchSource(sourceMode) {
    this.currentSource = sourceMode;

    if (sourceMode === "client_webcam") {
      // Start browser webcam access
      await this.startBrowserWebcam();
    } else {
      // Stop browser webcam if active
      this.stopBrowserWebcam();

      // Notify backend to switch to demo or cam0
      try {
        await fetch("/api/source/change", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: sourceMode }),
        });
      } catch (e) {
        console.warn("Could not notify backend of source change:", e);
      }

      if (sourceMode === "demo") {
        this.camBadge.textContent = "DEMO FEED";
        this.camSourceLabel.textContent = "CAM_01 — Manufacturing Floor / Assembly Zone";
      } else {
        this.camBadge.textContent = "CAM 0 LIVE";
        this.camSourceLabel.textContent = "CAM_00 — Server USB / Built-in Camera";
      }

      // Reconnect MJPEG stream
      this.videoFeed.src = `/api/video_feed?t=${Date.now()}`;
    }
  }

  async startBrowserWebcam() {
    this.camBadge.textContent = "MY WEBCAM LIVE";
    this.camSourceLabel.textContent = "CAM_CLIENT — Real-time Browser Webcam";

    try {
      // Request camera access from browser
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 25 } },
        audio: false,
      });

      this.webcamStream = stream;
      this.webcamVideo.srcObject = stream;
      await this.webcamVideo.play();

      // Setup canvas for frame extraction
      this.webcamCanvas.width = 640;
      this.webcamCanvas.height = 480;
      const ctx = this.webcamCanvas.getContext("2d");

      // Inform backend we are in client webcam mode
      await fetch("/api/source/change", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "client_webcam" }),
      });

      // Start frame capture and transmission loop (~20 FPS)
      this.webcamInterval = setInterval(() => {
        if (!this.webcamStream || this.isProcessingFrame) return;

        ctx.drawImage(this.webcamVideo, 0, 0, 640, 480);
        const dataUrl = this.webcamCanvas.toDataURL("image/jpeg", 0.75);

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.isProcessingFrame = true;
          this.ws.send(JSON.stringify({
            type: "client_frame",
            image: dataUrl,
          }));
        } else {
          // Fallback via HTTP POST
          this.isProcessingFrame = true;
          fetch("/api/pipeline/process_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl }),
          })
            .then((r) => r.json())
            .then((res) => {
              this.isProcessingFrame = false;
              if (res.annotated_image) {
                this.videoFeed.src = res.annotated_image;
              }
              if (res.metrics) {
                this.updateMetrics(res.metrics);
              }
              if (Array.isArray(res.alerts)) {
                res.alerts.forEach((a) => this.addAlert(a));
              }
            })
            .catch(() => {
              this.isProcessingFrame = false;
            });
        }
      }, 50);

      console.log("Browser Webcam live stream started successfully!");
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Could not access camera: " + err.message + "\nPlease check browser camera permissions.");
      // Fallback to demo
      const demoBtn = document.getElementById("btnSourceDemo");
      if (demoBtn) demoBtn.click();
    }
  }

  stopBrowserWebcam() {
    if (this.webcamInterval) {
      clearInterval(this.webcamInterval);
      this.webcamInterval = null;
    }
    if (this.webcamStream) {
      this.webcamStream.getTracks().forEach((track) => track.stop());
      this.webcamStream = null;
    }
    this.isProcessingFrame = false;
  }

  async syncPPEConfig() {
    const activePPE = [];
    document.querySelectorAll(".ppe-toggle-btn.active").forEach((btn) => {
      activePPE.push(btn.getAttribute("data-ppe"));
    });

    try {
      const resp = await fetch("/api/config/ppe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ required_ppe: activePPE }),
      });
      if (resp.ok) {
        console.log("PPE policy updated on edge backend:", activePPE);
      }
    } catch (e) {
      console.warn("Could not sync PPE policy:", e);
    }
  }

  async loadInitialStatus() {
    try {
      const resp = await fetch("/api/status");
      if (resp.ok) {
        const data = await resp.json();
        if (data.pipeline) {
          this.updateMetrics(data.pipeline);
        }
      }
    } catch (e) {
      // Backend starting up
    }
  }
}

// Instantiate dashboard on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new SafetyDashboard();
});
