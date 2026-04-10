#!/usr/bin/env python3
"""
Web服务器模块
提供管理界面和实时视频流
"""

import os
import time
import logging
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn


class WebServer:
    def __init__(self, config, camera_manager, ai_analyzer):
        self.config = config
        self.camera_manager = camera_manager
        self.ai_analyzer = ai_analyzer
        self.app = FastAPI(title="AI Camera Box")
        
        self.running = False
        self.thread = None
        
        ai_analyzer.set_camera_manager(camera_manager)
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            return self._get_index_html()
        
        @self.app.get("/api/status")
        async def get_status():
            return {
                'cameras': [
                    {'name': cam.name, 'source': cam.source, 'connected': cam.connected}
                    for cam in self.camera_manager.get_all_cameras()
                ],
                'ai_enabled': self.config.get('ai.enabled', True)
            }
        
        @self.app.get("/api/health")
        async def get_health():
            try:
                import psutil
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                return {'cpu': cpu, 'memory': memory, 'status': 'healthy'}
            except:
                return {'status': 'unknown'}
        
        @self.app.get("/stream/{camera_source:path}")
        async def video_stream(camera_source: str):
            return StreamingResponse(
                self._generate_frames(camera_source),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        @self.app.get("/stream")
        async def primary_video_stream():
            camera = self.camera_manager.get_primary_camera()
            if camera:
                return StreamingResponse(
                    self._generate_frames(camera.source),
                    media_type="multipart/x-mixed-replace; boundary=frame"
                )
            return Response("No camera available", status_code=404)

    def _get_index_html(self) -> str:
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI边缘计算摄像头分析盒子</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               background: #1a1a2e; color: #eee; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { text-align: center; padding: 30px 0; }
        header h1 { font-size: 2.5em; background: linear-gradient(45deg, #00d4ff, #7b2ff7); 
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status-bar { display: flex; justify-content: space-between; background: #16213e; 
                      padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; }
        .status-item { display: flex; align-items: center; gap: 10px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background: #4ade80; }
        .video-container { background: #0f0f23; border-radius: 15px; padding: 20px; 
                          box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
        .video-wrapper { position: relative; padding-bottom: 56.25%; background: #000; 
                         border-radius: 10px; overflow: hidden; }
        .video-wrapper img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                             object-fit: contain; }
        .no-camera { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                     text-align: center; color: #666; }
        .camera-selector { margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap; }
        .camera-btn { padding: 10px 20px; background: #16213e; border: 1px solid #334155; 
                     border-radius: 8px; color: #eee; cursor: pointer; transition: all 0.3s; }
        .camera-btn:hover, .camera-btn.active { background: #3b82f6; border-color: #3b82f6; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px; margin-top: 20px; }
        .info-card { background: #16213e; padding: 20px; border-radius: 10px; }
        .info-card h3 { color: #3b82f6; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI边缘计算摄像头分析盒子</h1>
            <p style="color: #64748b; margin-top: 10px;">即插即用 · 实时分析</p>
        </header>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>系统运行中</span>
            </div>
            <div class="status-item">
                <span id="cpu-status">CPU: --</span>
            </div>
            <div class="status-item">
                <span id="memory-status">内存: --</span>
            </div>
        </div>
        
        <div class="video-container">
            <div class="video-wrapper">
                <img id="video-stream" src="" alt="Video Stream">
                <div class="no-camera" id="no-camera">
                    <p>📷 等待摄像头连接...</p>
                    <p style="font-size: 0.9em; margin-top: 10px;">请插入USB摄像头或配置IP摄像头</p>
                </div>
            </div>
            <div class="camera-selector" id="camera-selector"></div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>📋 使用说明</h3>
                <p style="font-size: 0.9em; line-height: 1.8;">
                    1. 插入USB摄像头或配置IP摄像头<br>
                    2. 系统自动检测并开始分析<br>
                    3. 在此查看实时视频和检测结果<br>
                    4. 支持多摄像头切换
                </p>
            </div>
            <div class="info-card">
                <h3>🎯 检测功能</h3>
                <p style="font-size: 0.9em; line-height: 1.8;">
                    • 人物检测<br>
                    • 车辆检测<br>
                    • 物体识别<br>
                    • 人脸检测
                </p>
            </div>
            <div class="info-card">
                <h3>🔧 系统状态</h3>
                <p style="font-size: 0.9em; line-height: 1.8;" id="system-info">
                    正在加载...
                </p>
            </div>
        </div>
    </div>
    
    <script>
        let currentCamera = null;
        
        async function updateStatus() {
            try {
                const healthRes = await fetch('/api/health');
                const health = await healthRes.json();
                document.getElementById('cpu-status').textContent = `CPU: ${health.cpu}%`;
                document.getElementById('memory-status').textContent = `内存: ${health.memory}%`;
                
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();
                
                const selector = document.getElementById('camera-selector');
                selector.innerHTML = '';
                
                if (status.cameras.length > 0) {
                    document.getElementById('no-camera').style.display = 'none';
                    
                    status.cameras.forEach((cam, index) => {
                        const btn = document.createElement('button');
                        btn.className = 'camera-btn' + (currentCamera === cam.source ? ' active' : '');
                        btn.textContent = cam.name;
                        btn.onclick = () => selectCamera(cam.source);
                        selector.appendChild(btn);
                    });
                    
                    if (!currentCamera) {
                        selectCamera(status.cameras[0].source);
                    }
                    
                    document.getElementById('system-info').innerHTML = 
                        `摄像头数量: ${status.cameras.length}<br>` +
                        `AI分析: ${status.ai_enabled ? '已启用' : '已禁用'}`;
                } else {
                    document.getElementById('no-camera').style.display = 'block';
                    document.getElementById('video-stream').src = '';
                    document.getElementById('system-info').textContent = '等待摄像头连接...';
                }
            } catch (e) {
                console.error('Status update error:', e);
            }
        }
        
        function selectCamera(source) {
            currentCamera = source;
            document.getElementById('video-stream').src = `/stream/${encodeURIComponent(source)}`;
            updateStatus();
        }
        
        updateStatus();
        setInterval(updateStatus, 3000);
    </script>
</body>
</html>
"""

    def _generate_frames(self, camera_source: str):
        while True:
            camera = self.camera_manager.get_camera(camera_source)
            if camera:
                frame = camera.get_frame()
                if frame is not None:
                    if self.ai_analyzer:
                        results = self.ai_analyzer.get_results(camera_source)
                        frame = self.ai_analyzer.visualize(frame.copy(), results)
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
            time.sleep(0.033)

    def _run_server(self):
        host = self.config.get('web.host', '0.0.0.0')
        port = self.config.get('web.port', 8080)
        
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False
            )
        except Exception as e:
            logging.error(f"Web server error: {e}")

    def start(self):
        logging.info("Starting web server...")
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        host = self.config.get('web.host', '0.0.0.0')
        port = self.config.get('web.port', 8080)
        logging.info(f"Web server started at http://{host}:{port}")

    def stop(self):
        logging.info("Stopping web server...")
        self.running = False
