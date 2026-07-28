import os
import cv2
import json
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import insightface
from insightface.app import FaceAnalysis

# Inisialisasi FastAPI
app = FastAPI(title="AI Face Recognition & Anti-Spoofing Service")

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env file
load_dotenv()

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [orig.strip() for orig in allowed_origins_str.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_class=HTMLResponse)
async def webcam_tester():
    html_content = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face ID Real-Time Tester</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Outfit', sans-serif;
                background: radial-gradient(circle at center, #1e1e2f 0%, #0f0f1a 100%);
                color: #ffffff;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                justify-content: center;
            }
            .container {
                display: flex;
                flex-direction: row;
                gap: 30px;
                max-width: 1000px;
                width: 90%;
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 30px;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            }
            .camera-box {
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                position: relative;
            }
            .video-wrapper {
                position: relative;
                width: 100%;
                max-width: 480px;
                border-radius: 16px;
                overflow: hidden;
                border: 3px solid #3b82f6;
                box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
            }
            video {
                width: 100%;
                height: auto;
                display: block;
                transform: scaleX(-1); /* Mirror effect */
            }
            .scanner-line {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: linear-gradient(to right, transparent, #3b82f6, transparent);
                animation: scan 2s linear infinite;
                z-index: 10;
                box-shadow: 0 0 10px #3b82f6;
            }
            .countdown-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.6);
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 80px;
                font-weight: 800;
                color: #3b82f6;
                z-index: 20;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s ease;
            }
            .countdown-overlay.active {
                opacity: 1;
            }
            @keyframes scan {
                0% { top: 0%; }
                50% { top: 100%; }
                100% { top: 0%; }
            }
            .result-box {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 16px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                max-height: 480px;
                overflow-y: auto;
            }
            h1 {
                margin-top: 0;
                font-size: 24px;
                font-weight: 800;
                background: linear-gradient(to right, #3b82f6, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 24px;
            }
            button {
                margin-top: 20px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                border: none;
                color: white;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 30px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
            }
            button:active {
                transform: translateY(0);
            }
            pre {
                font-family: monospace;
                font-size: 12px;
                color: #34d399;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin: 0;
            }
            .status {
                margin-top: 10px;
                font-size: 16px;
                font-weight: 600;
                color: #9ca3af;
                min-height: 24px;
                text-align: center;
            }
            .step-indicator {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            .step-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
            }
            .step-dot.active {
                background: #3b82f6;
                box-shadow: 0 0 10px #3b82f6;
            }
            .step-dot.completed {
                background: #34d399;
            }
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <h1>Face ID Real-Time Enrollment Tester</h1>
        <div class="container">
            <div class="camera-box">
                <div class="step-indicator">
                    <div class="step-dot" id="dot-1"></div>
                    <div class="step-dot" id="dot-2"></div>
                    <div class="step-dot" id="dot-3"></div>
                </div>
                <div class="video-wrapper">
                    <div class="scanner-line"></div>
                    <div class="countdown-overlay" id="countdown">3</div>
                    <video id="webcam" autoplay playsinline></video>
                </div>
                <button id="start-btn">Mulai Pendaftaran Wajah</button>
                <div class="status" id="status-msg">Menginisialisasi kamera...</div>
            </div>
            <div class="result-box">
                <h3 style="margin-top:0; color:#3b82f6;">Status Pendaftaran:</h3>
                <pre id="api-result">Klik tombol untuk memulai proses pengambilan 3 sampel wajah (Depan -> Kanan -> Kiri).</pre>
            </div>
        </div>

        <script>
            const video = document.getElementById('webcam');
            const startBtn = document.getElementById('start-btn');
            const statusMsg = document.getElementById('status-msg');
            const apiResult = document.getElementById('api-result');
            
            const dots = [
                document.getElementById('dot-1'),
                document.getElementById('dot-2'),
                document.getElementById('dot-3')
            ];

            let embeddings = [];
            let currentStep = 0; // 0: Idle, 1: Depan, 2: Kanan, 3: Kiri, 4: Selesai
            let isDetecting = false;

            // Jalankan Kamera
            async function startWebcam() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    video.srcObject = stream;
                    statusMsg.innerText = "Kamera aktif. Silakan klik tombol untuk memulai.";
                    statusMsg.style.color = "#34d399";
                } catch (err) {
                    console.error("Error akses kamera: ", err);
                    statusMsg.innerText = "Gagal mengakses kamera. Pastikan izin kamera diberikan.";
                    statusMsg.style.color = "#f87171";
                }
            }

            // Loop capture sequential
            async function captureAndAnalyze() {
                if (!isDetecting || currentStep < 1 || currentStep > 3) {
                    isDetecting = false;
                    return;
                }

                // Buat frame capture dengan resolusi lebih kecil (320x240) untuk hemat bandwidth & CPU hosting
                const canvas = document.createElement('canvas');
                canvas.width = 320;
                canvas.height = 240;
                const ctx = canvas.getContext('2d');
                
                // Mirror
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Kompresi kualitas JPEG ke 0.8 untuk ukuran file jauh lebih kecil
                canvas.toBlob(async (blob) => {
                    if (!blob) {
                        if (isDetecting && currentStep >= 1 && currentStep <= 3) {
                            setTimeout(captureAndAnalyze, 200);
                        }
                        return;
                    }

                    const formData = new FormData();
                    formData.append('file', blob, 'face_stream.jpg');

                    try {
                        const response = await fetch('/analyze-face', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        
                        if (data.success) {
                            const detectedOrientation = data.pose.orientation;
                            const yaw = data.pose.yaw.toFixed(1);
                            
                            if (currentStep === 1) {
                                statusMsg.innerText = `[Langkah 1] Hadap depan... (Deteksi yaw: ${yaw}, pose: ${detectedOrientation})`;
                                statusMsg.style.color = "#3b82f6";
                                
                                if (detectedOrientation === "front") {
                                    embeddings.push(data.embedding);
                                    apiResult.innerText = "Sampel 1 (Depan) berhasil diambil secara otomatis!\nMenunggu Anda hadap kanan...";
                                    currentStep = 2;
                                    updateIndicators();
                                }
                            } else if (currentStep === 2) {
                                statusMsg.innerText = `[Langkah 2] Silakan HADAP KANAN... (Deteksi yaw: ${yaw}, pose: ${detectedOrientation})`;
                                statusMsg.style.color = "#f59e0b";
                                
                                if (detectedOrientation === "right") {
                                    embeddings.push(data.embedding);
                                    apiResult.innerText = "Sampel 2 (Kanan) berhasil diambil secara otomatis!\nMenunggu Anda hadap kiri...";
                                    currentStep = 3;
                                    updateIndicators();
                                }
                            } else if (currentStep === 3) {
                                statusMsg.innerText = `[Langkah 3] Silakan HADAP KIRI... (Deteksi yaw: ${yaw}, pose: ${detectedOrientation})`;
                                statusMsg.style.color = "#8b5cf6";
                                
                                if (detectedOrientation === "left") {
                                    embeddings.push(data.embedding);
                                    isDetecting = false;
                                    currentStep = 4;
                                    finishEnrollment();
                                    return;
                                }
                            }
                        } else {
                            statusMsg.innerText = `Wajah tidak stabil/tidak terdeteksi: ${data.error || 'Ubah pencahayaan'}`;
                            statusMsg.style.color = "#f87171";
                        }
                    } catch (err) {
                        console.error("Polling error: ", err);
                    }

                    // Panggil iterasi berikutnya jika proses deteksi masih aktif
                    if (isDetecting && currentStep >= 1 && currentStep <= 3) {
                        setTimeout(captureAndAnalyze, 200); // Jeda 200ms setelah request sebelumnya benar-benar selesai
                    }
                }, 'image/jpeg', 0.8);
            }

            // Mulai Loop Deteksi Pose Otomatis
            function startPoseDetection() {
                embeddings = [];
                currentStep = 1;
                updateIndicators();
                
                apiResult.innerText = "Mulai mendeteksi... Silakan luruskan wajah Anda menghadap ke depan.";
                isDetecting = true;
                captureAndAnalyze();
            }

            // Update Tampilan Dots
            function updateIndicators() {
                dots.forEach((dot, index) => {
                    dot.className = "step-dot";
                    if (index + 1 < currentStep) dot.classList.add('completed');
                    if (index + 1 === currentStep) dot.classList.add('active');
                });
            }

            // Selesaikan Pendaftaran
            function finishEnrollment() {
                statusMsg.innerText = "Pendaftaran Wajah Berhasil!";
                statusMsg.style.color = "#34d399";
                
                dots.forEach(dot => {
                    dot.className = "step-dot completed";
                });

                // Rata-ratakan 512 dimensi vektor
                const size = 512;
                const averagedEmbedding = new Array(size).fill(0);
                for (let i = 0; i < size; i++) {
                    let sum = 0;
                    for (const emb of embeddings) {
                        sum += emb[i];
                    }
                    averagedEmbedding[i] = sum / embeddings.length;
                }

                apiResult.innerText = JSON.stringify({
                    success: true,
                    message: "Registrasi Face ID Otomatis Berhasil!",
                    samples_captured: embeddings.length,
                    averaged_embedding: averagedEmbedding
                }, null, 2);

                startBtn.innerText = "Daftar Ulang";
                startBtn.disabled = false;
            }

            // Start Btn Event
            startBtn.addEventListener('click', () => {
                isDetecting = false;
                startBtn.disabled = true;
                startBtn.innerText = "Mendeteksi...";
                startPoseDetection();
            });

            // Start webcam
            startWebcam();
        </script>

    </body>
    </html>
    """
    return html_content



# Muat model InsightFace (Buffalo_L)
try:
    # Menggunakan CPU provider agar ramah terhadap resource shared hosting cPanel
    face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))
except Exception as e:
    print(f"Gagal memuat model InsightFace: {e}")

def check_liveness(image_np) -> float:
    """
    MiniFASNet Liveness Test (Mocking Liveness Score)
    Di produksi, silakan hubungkan dengan weights model MiniFASNet.
    Kembalikan skor antara 0 s.d 1. (Skor > 0.85 = Asli).
    """
    # Mocking: Menganggap wajah selalu asli (0.95) jika terdeteksi dengan baik.
    # Anda dapat mengganti ini dengan evaluasi model MiniFASNet (.pth)
    return 0.95

def calculate_cosine_similarity(v1, v2):
    """
    Menghitung Cosine Similarity antara dua array vektor menggunakan NumPy
    Formula: dot(A, B) / (||A|| * ||B||)
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """
    Endpoint saat pendaftaran (Enrollment) dengan deteksi pose real-time.
    Mengekstrak embedding & mendeteksi apakah wajah menghadap DEPAN, KANAN, atau KIRI.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"success": False, "error": "Format file gambar tidak valid."}

        # Liveness check
        liveness_score = check_liveness(img)
        if liveness_score < 0.85:
            return {
                "success": False,
                "error": "Liveness check failed. Spoofing terdeteksi.",
                "liveness_score": float(liveness_score)
            }

        # Deteksi Wajah
        faces = face_app.get(img)
        if not faces:
            return {"success": False, "error": "Wajah tidak terdeteksi pada gambar."}
        
        # Ambil wajah terbesar
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        embedding = face.normed_embedding.tolist()

        # Deteksi Pose (Yaw, Pitch, Roll)
        pitch, yaw, roll = 0.0, 0.0, 0.0
        if hasattr(face, 'pose') and face.pose is not None:
            pitch, yaw, roll = [float(x) for x in face.pose]
        else:
            # Fallback menggunakan keypoints mata dan hidung
            kps = face.kps
            eye_dist = float(np.linalg.norm(kps[1] - kps[0]))
            if eye_dist > 0:
                eye_center_x = float((kps[0][0] + kps[1][0]) / 2.0)
                nose_offset = float(kps[2][0] - eye_center_x)
                # Estimasi yaw (sensitivitas rasio hidung terhadap jarak mata)
                yaw = (nose_offset / eye_dist) * 110.0

        # Menentukan Kategori Pose Wajah (Depan, Kiri, Kanan)
        # Menyesuaikan dengan tampilan kamera mirror:
        # yaw < -18: Hadap Kiri (dari sudut pandang user)
        # yaw > 18: Hadap Kanan (dari sudut pandang user)
        # abs(yaw) < 12: Hadap Depan
        orientation = "unknown"
        if abs(yaw) < 12 and abs(pitch) < 15:
            orientation = "front"
        elif yaw < -18:
            orientation = "left"
        elif yaw > 18:
            orientation = "right"

        return {
            "success": True,
            "liveness_score": float(liveness_score),
            "embedding": embedding,
            "pose": {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "orientation": orientation
            }
        }

    except Exception as e:
        return {"success": False, "error": f"Kesalahan internal: {str(e)}"}

@app.post("/compare-face")
async def compare_face(
    file: UploadFile = File(...),
    registered_embedding_json: str = Form(...)
):
    """
    Endpoint saat verifikasi presensi (Verification).
    Membandingkan foto baru dengan array embedding wajah terdaftar dari PostgreSQL.
    """
    try:
        # Parse embedding terdaftar
        try:
            registered_emb = np.array(json.loads(registered_embedding_json), dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="Format registered_embedding_json tidak valid.")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Format file gambar baru tidak valid.")

        # Liveness check
        liveness_score = check_liveness(img)
        if liveness_score < 0.85:
            return {
                "success": False,
                "error": "Liveness check failed. Terdeteksi spoofing.",
                "liveness_score": float(liveness_score)
            }

        # Deteksi wajah baru
        faces = face_app.get(img)
        if not faces:
            raise HTTPException(status_code=400, detail="Wajah tidak terdeteksi pada foto verifikasi.")
        
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        new_emb = faces[0].normed_embedding

        # Hitung kemiripan
        similarity = calculate_cosine_similarity(registered_emb, new_emb)

        return {
            "success": True,
            "similarity": similarity,
            "liveness_score": float(liveness_score)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
