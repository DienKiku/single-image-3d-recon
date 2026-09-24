"""Three.js viewer for Streamlit."""
import json

def generate_viewer_html(
    width: int = 800,
    height: int = 600,
    background_color: str = "#1a1a2e",
    show_grid: bool = True,
    show_axes: bool = True,
) -> str:
    """Generate self-contained HTML for Three.js viewer."""
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: {background_color}; }}
            #info {{ position: absolute; top: 10px; right: 10px; color: white; font-family: sans-serif; pointer-events: none; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 4px; z-index: 100; }}
            canvas {{ display: block; width: 100vw; height: 100vh; }}
        </style>
        <script type="importmap">
          {{
            "imports": {{
              "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
              "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }}
          }}
        </script>
    </head>
    <body>
        <div id="info">
            <strong>3D Studio</strong><br/>
            Left-click: Orbit<br/>
            Right-click: Pan<br/>
            Scroll: Zoom
        </div>
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            const scene = new THREE.Scene();
            scene.background = new THREE.Color('{background_color}');

            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(5, 5, 5);
            camera.lookAt(0, 0, 0);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = 1.0;
            controls.maxDistance = 60.0;

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 7);
            scene.add(directionalLight);

            if ({str(show_grid).lower()}) {{
                const gridHelper = new THREE.GridHelper(10, 10, 0x444466, 0x444466);
                scene.add(gridHelper);
            }}

            if ({str(show_axes).lower()}) {{
                const axesHelper = new THREE.AxesHelper(3);
                scene.add(axesHelper);
            }}

            const userMeshes = [];

            window.addMesh = function(geometryType, params) {{
                let geometry, material, mesh;
                material = new THREE.MeshStandardMaterial({{ color: 0x88ccff }});
                if (geometryType === 'box') {{
                    geometry = new THREE.BoxGeometry(params.w || 1, params.h || 1, params.d || 1);
                }} else if (geometryType === 'sphere') {{
                    geometry = new THREE.SphereGeometry(params.r || 1, 32, 32);
                }} else {{
                    geometry = new THREE.BoxGeometry(1, 1, 1);
                }}
                mesh = new THREE.Mesh(geometry, material);
                mesh.userData.basePosition = new THREE.Vector3(params.x || 0, params.y || 0, params.z || 0);
                mesh.userData.explosionDir = new THREE.Vector3(params.ex || 0, params.ey || 0, params.ez || 0);
                mesh.position.copy(mesh.userData.basePosition);
                scene.add(mesh);
                userMeshes.push(mesh);
            }};

            window.updateExplosion = function(factor) {{
                userMeshes.forEach(mesh => {{
                    if (mesh.userData.basePosition && mesh.userData.explosionDir) {{
                        const offset = mesh.userData.explosionDir.clone().multiplyScalar(factor);
                        mesh.position.copy(mesh.userData.basePosition).add(offset);
                    }}
                }});
            }};

            window.clearScene = function() {{
                userMeshes.forEach(mesh => {{
                    scene.remove(mesh);
                    if (mesh.geometry) mesh.geometry.dispose();
                    if (mesh.material) mesh.material.dispose();
                }});
                userMeshes.length = 0;
            }};

            window.addEventListener('message', (event) => {{
                const data = event.data;
                if (!data || !data.action) return;
                
                if (data.action === 'addMesh') {{
                    window.addMesh(data.geometryType, data.params || {{}});
                }} else if (data.action === 'updateExplosion') {{
                    window.updateExplosion(data.factor || 0);
                }} else if (data.action === 'clearScene') {{
                    window.clearScene();
                }}
            }});

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // For Streamlit static load sync
            window.addEventListener('load', () => {{
                // Read from initial hash or similar if needed in the future
            }});
        </script>
    </body>
    </html>
    """
    return html

def generate_placeholder_scene_html(width: int = 800, height: int = 600) -> str:
    """Generate simple HTML for empty state placeholder."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: #1a1a2e;
                color: #88ccff;
                font-family: sans-serif;
            }}
            .icon {{ font-size: 48px; margin-bottom: 20px; }}
            .message {{ font-size: 18px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="icon">🧊</div>
        <div class="message">Upload an image and generate 3D layers to begin</div>
    </body>
    </html>
    """

def generate_viewer_with_meshes_html(
    mesh_data: list[dict],
    explosion_factor: float = 0.0,
    width: int = 800,
    height: int = 600,
    background_color: str = "#1a1a2e",
    show_grid: bool = True,
    show_axes: bool = True,
) -> str:
    """Generate self-contained HTML for Three.js viewer with embedded mesh data."""
    mesh_data_json = json.dumps(mesh_data, separators=(',', ':'))
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: {background_color}; }}
            #info {{ position: absolute; top: 10px; right: 10px; color: white; font-family: sans-serif; pointer-events: none; background: rgba(18,20,36,0.85); padding: 8px 12px; border-radius: 6px; z-index: 100; text-align: right; box-shadow: 0 4px 12px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); }}
            #layer-panel {{ position: absolute; top: 10px; left: 10px; color: white; font-family: sans-serif; background: rgba(18,20,36,0.85); padding: 10px 14px; border-radius: 6px; z-index: 100; max-height: 90vh; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); }}
            .layer-item {{ display: flex; align-items: center; margin-bottom: 5px; cursor: pointer; padding: 3px 6px; user-select: none; border-radius: 3px; }}
            .layer-item:hover {{ background: rgba(255,255,255,0.15); }}
            .layer-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; display: inline-block; }}
            #tooltip {{ position: absolute; background: rgba(10,12,25,0.9); color: white; padding: 6px 12px; border-radius: 4px; font-family: sans-serif; font-size: 12px; pointer-events: none; display: none; z-index: 200; white-space: nowrap; border: 1px solid rgba(255,255,255,0.15); }}
            .hud-btn {{
                background: #232845;
                color: #88ccff;
                border: 1px solid #4a5482;
                padding: 4px 9px;
                border-radius: 4px;
                font-size: 11px;
                cursor: pointer;
                pointer-events: auto;
                transition: all 0.15s ease;
            }}
            .hud-btn:hover {{
                background: #333d6b;
                border-color: #7085c9;
            }}
            .hud-btn.active {{
                background: #3b5998;
                color: #ffffff;
                border-color: #88b0ff;
                font-weight: bold;
            }}
            canvas {{ display: block; width: 100vw; height: 100vh; }}
        </style>
        <script type="importmap">
          {{
            "imports": {{
              "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
              "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }}
          }}
        </script>
    </head>
    <body>
        <div id="info">
            <div style="font-weight: bold; font-size: 13px; margin-bottom: 3px; color: #aaccff;">🧊 3D Studio Viewer</div>
            <div style="font-size: 11px; opacity: 0.85; margin-bottom: 6px;"><span id="mesh-count">0</span> meshes loaded</div>
            <div style="display: flex; gap: 4px; justify-content: flex-end; flex-wrap: wrap; margin-bottom: 4px;">
                <button id="btn-mode-clay" class="hud-btn active">🎨 Studio Clay</button>
                <button id="btn-mode-textured" class="hud-btn">🖼️ Ảnh thật</button>
                <button id="btn-mode-wireframe" class="hud-btn">📐 Lưới</button>
            </div>
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; margin-bottom: 4px;">
                    <span style="color: #aaccff; font-weight: 500;">🔍 Thu phóng (Zoom):</span>
                    <span id="zoom-pct-display" style="font-weight: bold; color: #ffffff; background: #232845; padding: 1px 6px; border-radius: 3px; border: 1px solid #4a5482;">100%</span>
                </div>
                <div style="display: flex; gap: 4px; align-items: center;">
                    <button id="btn-zoom-out" class="hud-btn" style="flex: 0 0 32px; padding: 2px 0; font-weight: bold;" title="Thu nhỏ 1%">-1%</button>
                    <input type="range" id="zoom-slider" min="25" max="500" value="100" step="1" style="flex: 1; accent-color: #5588ff; cursor: pointer;" title="Kéo để chỉnh zoom theo %">
                    <button id="btn-zoom-in" class="hud-btn" style="flex: 0 0 32px; padding: 2px 0; font-weight: bold;" title="Phóng to 1%">+1%</button>
                </div>
            </div>
            <div>
                <button id="btn-reset-view" class="hud-btn" style="width: 100%; margin-top: 6px;">🎥 Căn lại góc nhìn (Reset View: 100%)</button>
            </div>
        </div>
        <div id="layer-panel">
            <strong style="font-size: 13px; color: #aaccff;">Components</strong><br/>
            <div id="layer-list" style="margin-top: 6px;"></div>
        </div>
        <div id="tooltip"></div>
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            const scene = new THREE.Scene();
            scene.background = new THREE.Color('{background_color}');

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 100000);
            camera.position.set(0, 0, 300);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.enablePan = true;
            controls.enableZoom = false; // Handled by controlled 1% zoom system

            // Studio 4-Point Lighting Rig for Rich Volumetric Shading
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
            scene.add(ambientLight);

            const keyLight = new THREE.DirectionalLight(0xffffff, 0.90);
            keyLight.position.set(80, 120, 100);
            scene.add(keyLight);

            const fillLight = new THREE.DirectionalLight(0x99bbee, 0.45);
            fillLight.position.set(-80, -40, -60);
            scene.add(fillLight);

            const rimLight = new THREE.DirectionalLight(0xffeedd, 0.60);
            rimLight.position.set(0, 100, -120);
            scene.add(rimLight);

            const frontLight = new THREE.DirectionalLight(0xffffff, 0.35);
            frontLight.position.set(0, 20, 150);
            scene.add(frontLight);

            if ({str(show_grid).lower()}) {{
                const gridHelper = new THREE.GridHelper(10, 10, 0x444466, 0x444466);
                scene.add(gridHelper);
            }}

            if ({str(show_axes).lower()}) {{
                const axesHelper = new THREE.AxesHelper(3);
                scene.add(axesHelper);
            }}

            const userMeshes = [];
            
            const meshData = {mesh_data_json};
            const initialExplosion = {explosion_factor};

            document.getElementById('mesh-count').textContent = meshData.length;

            const layerList = document.getElementById('layer-list');
            const boundingBox = new THREE.Box3();
            let hasMeshes = false;

            meshData.forEach((data, index) => {{
                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(data.vertices);
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                geometry.setIndex(data.faces);
                geometry.computeVertexNormals();
                
                if (data.uvs && data.uvs.length > 0) {{
                    const uvs = new Float32Array(data.uvs);
                    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
                }}
                if (data.vertexColors && data.vertexColors.length > 0) {{
                    const colors = new Float32Array(data.vertexColors);
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                }}
                
                // 1. Studio Clay Material (Matte Off-White CAD Sculpture)
                const clayMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xdedee4,
                    roughness: 0.45,
                    metalness: 0.05,
                    side: THREE.DoubleSide
                }});

                // 2. Textured Material (Real Photo Textures or Colors)
                let texturedMaterial;
                if (data.textureDataUri && data.uvs && data.uvs.length > 0) {{
                    const texLoader = new THREE.TextureLoader();
                    const texture = texLoader.load(data.textureDataUri);
                    texture.colorSpace = THREE.SRGBColorSpace;
                    texture.minFilter = THREE.LinearMipmapLinearFilter;
                    texture.magFilter = THREE.LinearFilter;
                    texture.generateMipmaps = true;
                    if (renderer.capabilities) {{
                        texture.anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy());
                    }}
                    texturedMaterial = new THREE.MeshStandardMaterial({{
                        map: texture,
                        roughness: 0.45,
                        metalness: 0.05,
                        side: THREE.DoubleSide
                    }});
                }} else if (data.vertexColors && data.vertexColors.length > 0) {{
                    texturedMaterial = new THREE.MeshStandardMaterial({{
                        vertexColors: true,
                        roughness: 0.70,
                        metalness: 0.05,
                        side: THREE.DoubleSide
                    }});
                }} else {{
                    texturedMaterial = new THREE.MeshStandardMaterial({{
                        color: data.color,
                        roughness: 0.65,
                        metalness: 0.05,
                        side: THREE.DoubleSide
                    }});
                }}
                
                // Default to Clay Material (Crisp 3D form shading just like reference model)
                const mesh = new THREE.Mesh(geometry, clayMaterial);
                mesh.userData.clayMaterial = clayMaterial;
                mesh.userData.texturedMaterial = texturedMaterial;
                mesh.userData.basePosition = new THREE.Vector3(...data.basePosition);
                mesh.userData.explosionDir = new THREE.Vector3(...data.explosionDir);
                mesh.userData.layerId = data.layerId;
                mesh.userData.dimensionsMm = data.dimensionsMm;
                
                const offset = mesh.userData.explosionDir.clone().multiplyScalar(initialExplosion);
                mesh.position.copy(mesh.userData.basePosition).add(offset);
                
                scene.add(mesh);
                userMeshes.push(mesh);
                
                mesh.updateMatrixWorld(true);
                const box = new THREE.Box3().setFromObject(mesh);
                if (!box.isEmpty()) {{
                    if (!hasMeshes) {{
                        boundingBox.copy(box);
                        hasMeshes = true;
                    }} else {{
                        boundingBox.union(box);
                    }}
                }}

                const layerItem = document.createElement('div');
                layerItem.className = 'layer-item';
                layerItem.innerHTML = `<span class="layer-color" style="background: ${{data.color}}"></span> ${{data.layerId}}`;
                layerItem.addEventListener('click', () => {{
                    const oldEmissive = mesh.material.emissive.clone();
                    mesh.material.emissive.setHex(0x555555);
                    setTimeout(() => {{
                        mesh.material.emissive.copy(oldEmissive);
                    }}, 300);
                }});
                layerList.appendChild(layerItem);
            }});

            let initialCameraPos = new THREE.Vector3();
            let initialTarget = new THREE.Vector3();
            let baseDistance = 300;
            let currentZoomPercent = 100;

            function updateZoom(newPercent) {{
                newPercent = Math.max(25, Math.min(500, Math.round(newPercent)));
                currentZoomPercent = newPercent;

                const slider = document.getElementById('zoom-slider');
                const display = document.getElementById('zoom-pct-display');
                if (slider && parseInt(slider.value) !== currentZoomPercent) {{
                    slider.value = currentZoomPercent;
                }}
                if (display) {{
                    display.textContent = currentZoomPercent + '%';
                }}

                if (controls.target) {{
                    const ray = camera.position.clone().sub(controls.target);
                    const currentDist = ray.length();
                    if (currentDist > 1e-4) {{
                        const dir = ray.normalize();
                        const targetDist = baseDistance / (currentZoomPercent / 100.0);
                        camera.position.copy(controls.target).add(dir.multiplyScalar(targetDist));
                        camera.updateProjectionMatrix();
                        controls.update();
                    }}
                }}
            }}

            // Slider input event (1% step)
            document.getElementById('zoom-slider')?.addEventListener('input', (e) => {{
                updateZoom(parseFloat(e.target.value));
            }});

            // Step buttons (-1% and +1%)
            document.getElementById('btn-zoom-out')?.addEventListener('click', () => {{
                updateZoom(currentZoomPercent - 1);
            }});
            document.getElementById('btn-zoom-in')?.addEventListener('click', () => {{
                updateZoom(currentZoomPercent + 1);
            }});

            // Controlled mouse wheel zoom (strictly 1% per wheel tick, prevents jumping to 1000x or 0.001x)
            renderer.domElement.addEventListener('wheel', (e) => {{
                e.preventDefault();
                e.stopPropagation();
                const delta = e.deltaY < 0 ? 5 : -5;
                updateZoom(currentZoomPercent + delta);
            }}, {{ passive: false }});

            if (hasMeshes) {{
                const center = new THREE.Vector3();
                boundingBox.getCenter(center);
                const size = new THREE.Vector3();
                boundingBox.getSize(size);
                
                const maxDim = Math.max(size.x, size.y, size.z, 1.0);
                const fov = camera.fov * (Math.PI / 180);
                let dist = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5;
                baseDistance = dist;
                currentZoomPercent = 100;
                
                // 3/4 Isometric perspective (elevated 25%, angled 45%)
                camera.position.set(
                    center.x + dist * 0.45,
                    center.y + dist * 0.25,
                    center.z + dist * 1.1
                );
                camera.near = Math.max(0.01, maxDim * 0.005);
                camera.far = 100000;
                camera.updateProjectionMatrix();
                
                controls.target.copy(center);
                controls.minDistance = Math.max(0.05, maxDim * 0.08);
                controls.maxDistance = Math.max(10000, maxDim * 50);
                controls.update();

                initialCameraPos.copy(camera.position);
                initialTarget.copy(center);
                updateZoom(100);
            }}

            document.getElementById('btn-reset-view')?.addEventListener('click', () => {{
                if (hasMeshes) {{
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialTarget);
                    controls.update();
                    updateZoom(100);
                }}
            }});

            let currentShader = 'clay';
            let isWireframe = false;

            const btnClay = document.getElementById('btn-mode-clay');
            const btnTextured = document.getElementById('btn-mode-textured');
            const btnWireframe = document.getElementById('btn-mode-wireframe');

            function applyShader(mode) {{
                currentShader = mode;
                userMeshes.forEach(mesh => {{
                    const mat = (mode === 'clay') ? mesh.userData.clayMaterial : mesh.userData.texturedMaterial;
                    mat.wireframe = isWireframe;
                    mesh.material = mat;
                }});
                if (btnClay) btnClay.className = (mode === 'clay') ? 'hud-btn active' : 'hud-btn';
                if (btnTextured) btnTextured.className = (mode === 'textured') ? 'hud-btn active' : 'hud-btn';
            }}

            btnClay?.addEventListener('click', () => applyShader('clay'));
            btnTextured?.addEventListener('click', () => applyShader('textured'));
            btnWireframe?.addEventListener('click', () => {{
                isWireframe = !isWireframe;
                if (btnWireframe) btnWireframe.className = isWireframe ? 'hud-btn active' : 'hud-btn';
                userMeshes.forEach(mesh => {{
                    mesh.material.wireframe = isWireframe;
                }});
            }});

            window.addMesh = function(geometryType, params) {{
                let geometry, material, mesh;
                material = new THREE.MeshStandardMaterial({{ color: 0x88ccff }});
                if (geometryType === 'box') {{
                    geometry = new THREE.BoxGeometry(params.w || 1, params.h || 1, params.d || 1);
                }} else if (geometryType === 'sphere') {{
                    geometry = new THREE.SphereGeometry(params.r || 1, 32, 32);
                }} else {{
                    geometry = new THREE.BoxGeometry(1, 1, 1);
                }}
                mesh = new THREE.Mesh(geometry, material);
                mesh.userData.basePosition = new THREE.Vector3(params.x || 0, params.y || 0, params.z || 0);
                mesh.userData.explosionDir = new THREE.Vector3(params.ex || 0, params.ey || 0, params.ez || 0);
                mesh.position.copy(mesh.userData.basePosition);
                scene.add(mesh);
                userMeshes.push(mesh);
            }};

            window.updateExplosion = function(factor) {{
                userMeshes.forEach(mesh => {{
                    if (mesh.userData.basePosition && mesh.userData.explosionDir) {{
                        const offset = mesh.userData.explosionDir.clone().multiplyScalar(factor);
                        mesh.position.copy(mesh.userData.basePosition).add(offset);
                    }}
                }});
            }};

            window.clearScene = function() {{
                userMeshes.forEach(mesh => {{
                    scene.remove(mesh);
                    if (mesh.geometry) mesh.geometry.dispose();
                    if (mesh.material) mesh.material.dispose();
                }});
                userMeshes.length = 0;
            }};

            window.addEventListener('message', (event) => {{
                const data = event.data;
                if (!data || !data.action) return;
                
                if (data.action === 'addMesh') {{
                    window.addMesh(data.geometryType, data.params || {{}});
                }} else if (data.action === 'updateExplosion') {{
                    window.updateExplosion(data.factor || 0);
                }} else if (data.action === 'clearScene') {{
                    window.clearScene();
                }}
            }});

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            const tooltip = document.getElementById('tooltip');

            window.addEventListener('mousemove', (event) => {{
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(userMeshes, false);
                
                if (intersects.length > 0) {{
                    const mesh = intersects[0].object;
                    let html = `Layer: ${{mesh.userData.layerId}}`;
                    if (mesh.userData.dimensionsMm) {{
                        const dims = mesh.userData.dimensionsMm;
                        html += `<br/>Dims: ${{dims[0].toFixed(1)}} x ${{dims[1].toFixed(1)}} x ${{dims[2].toFixed(1)}} mm`;
                    }}
                    tooltip.innerHTML = html;
                    tooltip.style.display = 'block';
                    tooltip.style.left = event.clientX + 15 + 'px';
                    tooltip.style.top = event.clientY + 15 + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }});

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    return html
