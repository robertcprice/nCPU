//! Three.js/3D Graphics Support for nCPU/nSynth
//!
//! 3D scene generation, geometry, materials, and rendering.

/// 3D scene
#[derive(Debug, Clone)]
pub struct Scene {
    /// Scene objects
    pub objects: Vec<Object3D>,
    /// Camera
    pub camera: Camera,
    /// Renderer config
    pub renderer: Renderer,
    /// Lights
    pub lights: Vec<Light>,
    /// Background
    pub background: Option<Color>,
}

/// 3D object
#[derive(Debug, Clone)]
pub struct Object3D {
    /// Object ID
    pub id: String,
    /// Geometry
    pub geometry: Geometry,
    /// Material
    pub material: Material,
    /// Position
    pub position: Vec3,
    /// Rotation
    pub rotation: Vec3,
    /// Scale
    pub scale: Vec3,
    /// Children
    pub children: Vec<Object3D>,
}

/// 3D vector
#[derive(Debug, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Create new vector
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

/// Geometry type
#[derive(Debug, Clone)]
pub enum Geometry {
    Box {
        width: f32,
        height: f32,
        depth: f32,
    },
    Sphere {
        radius: f32,
        width_segments: u32,
        height_segments: u32,
    },
    Plane {
        width: f32,
        height: f32,
    },
    Cylinder {
        radius_top: f32,
        radius_bottom: f32,
        height: f32,
        radial_segments: u32,
    },
    Cone {
        radius: f32,
        height: f32,
        radial_segments: u32,
    },
    Torus {
        radius: f32,
        tube: f32,
        radial_segments: u32,
        tubular_segments: u32,
    },
    Custom {
        vertices: Vec<Vec3>,
        indices: Vec<u32>,
    },
}

/// Material type
#[derive(Debug, Clone)]
pub enum Material {
    /// Basic material (not affected by light)
    Basic { color: Color },
    /// Lambert material (diffuse lighting)
    Lambert { color: Color },
    /// Phong material (shiny)
    Phong {
        color: Color,
        specular: Color,
        shininess: f32,
    },
    /// Standard material (PBR)
    Standard {
        color: Color,
        roughness: f32,
        metalness: f32,
    },
    /// Texture material
    Texture {
        texture: String,
        normal_map: Option<String>,
    },
}

/// Color (RGB)
#[derive(Debug, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    /// Create from RGB
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create from hex
    pub fn hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as u8,
            g: ((hex >> 8) & 0xFF) as u8,
            b: (hex & 0xFF) as u8,
        }
    }

    /// To CSS string
    pub fn to_css(&self) -> String {
        format!("rgb({}, {}, {})", self.r, self.g, self.b)
    }

    /// To hex string
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

/// Camera
#[derive(Debug, Clone)]
pub struct Camera {
    /// Camera type
    pub camera_type: CameraType,
    /// Field of view (degrees)
    pub fov: f32,
    /// Near clip distance
    pub near: f32,
    /// Far clip distance
    pub far: f32,
    /// Position
    pub position: Vec3,
    /// Look-at target
    pub target: Vec3,
}

/// Camera type
#[derive(Debug, Clone, Copy)]
pub enum CameraType {
    Perspective,
    Orthographic,
}

/// Renderer configuration
#[derive(Debug, Clone)]
pub struct Renderer {
    /// Canvas width
    pub width: u32,
    /// Canvas height
    pub height: u32,
    /// Enable shadows
    pub shadows: bool,
    /// Antialiasing
    pub antialias: bool,
    /// Pixel ratio
    pub pixel_ratio: f32,
}

/// Light
#[derive(Debug, Clone)]
pub enum Light {
    Ambient {
        color: Color,
        intensity: f32,
    },
    Directional {
        color: Color,
        intensity: f32,
        position: Vec3,
    },
    Point {
        color: Color,
        intensity: f32,
        position: Vec3,
        distance: f32,
        decay: f32,
    },
    Spot {
        color: Color,
        intensity: f32,
        position: Vec3,
        target: Vec3,
        angle: f32,
        penumbra: f32,
    },
}

impl Scene {
    /// Create new scene
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            objects: Vec::new(),
            camera: Camera {
                camera_type: CameraType::Perspective,
                fov: 75.0,
                near: 0.1,
                far: 1000.0,
                position: Vec3::new(0.0, 5.0, 10.0),
                target: Vec3::zero(),
            },
            renderer: Renderer {
                width,
                height,
                shadows: true,
                antialias: true,
                pixel_ratio: window_device_pixel_ratio(),
            },
            lights: Vec::new(),
            background: None,
        }
    }

    /// Add object to scene
    pub fn add(mut self, object: Object3D) -> Self {
        self.objects.push(object);
        self
    }

    /// Add light to scene
    pub fn add_light(mut self, light: Light) -> Self {
        self.lights.push(light);
        self
    }

    /// Set background color
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }

    /// Generate Three.js code
    pub fn to_threejs(&self) -> String {
        let mut code = String::new();

        // Imports
        code.push_str("import * as THREE from 'three';\n\n");

        // Scene setup
        code.push_str("const scene = new THREE.Scene();\n");

        // Camera
        code.push_str(&format!(
            "const camera = new THREE.{}Camera({}, {}, {}, {});\n",
            match self.camera.camera_type {
                CameraType::Perspective => "Perspective",
                CameraType::Orthographic => "Orthographic",
            },
            self.camera.fov,
            (self.renderer.width as f32) / self.renderer.pixel_ratio,
            (self.renderer.height as f32) / self.renderer.pixel_ratio,
            self.camera.near
        ));
        code.push_str(&format!(
            "camera.position.set({}, {}, {});\n",
            self.camera.position.x, self.camera.position.y, self.camera.position.z
        ));
        code.push_str("const renderer = new THREE.WebGLRenderer({ antialias: true });\n");
        code.push_str(&format!(
            "renderer.setSize({}, {});\n",
            self.renderer.width, self.renderer.height
        ));
        if self.renderer.shadows {
            code.push_str("renderer.shadowMap.enabled = true;\n");
        }

        // Lights
        for light in &self.lights {
            code.push_str(&self.light_to_threejs(light));
        }

        // Objects
        for object in &self.objects {
            code.push_str(&self.object_to_threejs(object, 2));
        }

        // Animation loop
        code.push_str("\nfunction animate() {\n");
        code.push_str("  requestAnimationFrame(animate);\n");
        code.push_str("  renderer.render(scene, camera);\n");
        code.push_str("}\n");
        code.push_str("\nanimate();\n");

        code
    }

    fn light_to_threejs(&self, light: &Light) -> String {
        match light {
            Light::Ambient { color, intensity } => {
                format!("const ambientLight = new THREE.AmbientLight(0x{}, {});\nscene.add(ambientLight);\n",
                    color.to_hex(), intensity)
            }
            Light::Directional {
                color,
                intensity,
                position,
            } => {
                format!("const dirLight = new THREE.DirectionalLight(0x{}, {});\ndirLight.position.set({}, {}, {});\nscene.add(dirLight);\n",
                    color.to_hex(), intensity, position.x, position.y, position.z)
            }
            Light::Point {
                color,
                intensity,
                position,
                distance,
                decay,
            } => {
                format!("const pointLight = new THREE.PointLight(0x{}, {}, {}, {});\npointLight.position.set({}, {}, {});\nscene.add(pointLight);\n",
                    color.to_hex(), intensity, distance, decay, position.x, position.y, position.z)
            }
            Light::Spot {
                color,
                intensity,
                position,
                target,
                angle,
                penumbra,
            } => {
                format!("const spotLight = new THREE.SpotLight(0x{}, {}, {}, {});\nspotLight.position.set({}, {}, {});\nspotLight.target.position.set({}, {}, {});\nscene.add(spotLight);\n",
                    color.to_hex(), intensity, angle, penumbra,
                    position.x, position.y, position.z, target.x, target.y, target.z)
            }
        }
    }

    fn object_to_threejs(&self, object: &Object3D, indent: usize) -> String {
        let spaces = " ".repeat(indent);
        let mut code = String::new();

        // Geometry
        code.push_str(&spaces);
        code.push_str(&format!("const {}_geom = ", object.id));
        code.push_str(&self.geometry_to_threejs(&object.geometry));

        // Material
        code.push_str(&spaces);
        code.push_str(&format!("const {}_mat = ", object.id));
        code.push_str(&self.material_to_threejs(&object.material));

        // Mesh
        code.push_str(&spaces);
        code.push_str(&format!(
            "const {} = new THREE.Mesh({}_geom, {}_mat);\n",
            object.id, object.id, object.id
        ));
        code.push_str(&spaces);
        code.push_str(&format!(
            "{}.position.set({}, {}, {});\n",
            object.id, object.position.x, object.position.y, object.position.z
        ));
        code.push_str(&spaces);
        code.push_str(&format!(
            "{}.rotation.set({}, {}, {});\n",
            object.id, object.rotation.x, object.rotation.y, object.rotation.z
        ));
        code.push_str(&spaces);
        code.push_str(&format!(
            "{}.scale.set({}, {}, {});\n",
            object.id, object.scale.x, object.scale.y, object.scale.z
        ));
        code.push_str(&spaces);
        code.push_str(&format!("scene.add({});\n", object.id));

        // Children
        for child in &object.children {
            code.push_str(&self.object_to_threejs(child, indent));
        }

        code
    }

    fn geometry_to_threejs(&self, geometry: &Geometry) -> String {
        match geometry {
            Geometry::Box {
                width,
                height,
                depth,
            } => {
                format!("new THREE.BoxGeometry({}, {}, {});\n", width, height, depth)
            }
            Geometry::Sphere {
                radius,
                width_segments,
                height_segments,
            } => {
                format!(
                    "new THREE.SphereGeometry({}, {}, {});\n",
                    radius, width_segments, height_segments
                )
            }
            Geometry::Plane { width, height } => {
                format!("new THREE.PlaneGeometry({}, {});\n", width, height)
            }
            Geometry::Cylinder {
                radius_top,
                radius_bottom,
                height,
                radial_segments,
            } => {
                format!(
                    "new THREE.CylinderGeometry({}, {}, {}, {});\n",
                    radius_top, radius_bottom, height, radial_segments
                )
            }
            Geometry::Cone {
                radius,
                height,
                radial_segments,
            } => {
                format!(
                    "new THREE.ConeGeometry({}, {}, {});\n",
                    radius, height, radial_segments
                )
            }
            Geometry::Torus {
                radius,
                tube,
                radial_segments,
                tubular_segments,
            } => {
                format!(
                    "new THREE.TorusGeometry({}, {}, {}, {});\n",
                    radius, tube, radial_segments, tubular_segments
                )
            }
            Geometry::Custom { .. } => String::from("/* custom geometry */\n"),
        }
    }

    fn material_to_threejs(&self, material: &Material) -> String {
        match material {
            Material::Basic { color } => {
                format!(
                    "new THREE.MeshBasicMaterial({{ color: 0x{} }});\n",
                    color.to_hex()
                )
            }
            Material::Lambert { color } => {
                format!(
                    "new THREE.MeshLambertMaterial({{ color: 0x{} }});\n",
                    color.to_hex()
                )
            }
            Material::Phong {
                color,
                specular,
                shininess,
            } => {
                format!("new THREE.MeshPhongMaterial({{ color: 0x{}, specular: 0x{}, shininess: {} }});\n",
                    color.to_hex(), specular.to_hex(), shininess)
            }
            Material::Standard {
                color,
                roughness,
                metalness,
            } => {
                format!("new THREE.MeshStandardMaterial({{ color: 0x{}, roughness: {}, metalness: {} }});\n",
                    color.to_hex(), roughness, metalness)
            }
            Material::Texture {
                texture,
                normal_map,
            } => {
                let mut s = format!(
                    "new THREE.MeshStandardMaterial({{ map: THREE.TextureLoader.load('{}')",
                    texture
                );
                if let Some(normal) = normal_map {
                    s.push_str(&format!(
                        ", normalMap: THREE.TextureLoader.load('{}')",
                        normal
                    ));
                }
                s.push_str(" }});\n");
                s
            }
        }
    }
}

/// Get window device pixel ratio (would be from browser in real usage)
fn window_device_pixel_ratio() -> f32 {
    1.0
}

/// Geometry builder
#[derive(Debug, Clone)]
pub struct GeometryBuilder {
    geometry: Option<Geometry>,
}

impl GeometryBuilder {
    pub fn new() -> Self {
        Self { geometry: None }
    }

    pub fn box_(mut self, width: f32, height: f32, depth: f32) -> Self {
        self.geometry = Some(Geometry::Box {
            width,
            height,
            depth,
        });
        self
    }

    pub fn sphere(mut self, radius: f32) -> Self {
        self.geometry = Some(Geometry::Sphere {
            radius,
            width_segments: 32,
            height_segments: 16,
        });
        self
    }

    pub fn build(self) -> Geometry {
        self.geometry.unwrap_or(Geometry::Box {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        })
    }
}

/// Preset scenes
pub struct PresetScenes;

impl PresetScenes {
    /// Simple rotating cube scene
    pub fn rotating_cube() -> Scene {
        Scene::new(800, 600)
            .add_light(Light::Ambient {
                color: Color::rgb(128, 128, 128),
                intensity: 0.5,
            })
            .add_light(Light::Directional {
                color: Color::rgb(255, 255, 255),
                intensity: 1.0,
                position: Vec3::new(5.0, 10.0, 7.5),
            })
            .add(Object3D {
                id: "cube".to_string(),
                geometry: Geometry::Box {
                    width: 1.0,
                    height: 1.0,
                    depth: 1.0,
                },
                material: Material::Phong {
                    color: Color::hex(0x00ff00),
                    specular: Color::hex(0x111111),
                    shininess: 100.0,
                },
                position: Vec3::zero(),
                rotation: Vec3::zero(),
                scale: Vec3::new(1.0, 1.0, 1.0),
                children: Vec::new(),
            })
    }

    /// Solar system scene
    pub fn solar_system() -> Scene {
        Scene::new(800, 600)
            .with_background(Color::hex(0x000000))
            .add_light(Light::Point {
                color: Color::rgb(255, 255, 200),
                intensity: 2.0,
                position: Vec3::zero(),
                distance: 1000.0,
                decay: 0.0,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_creation() {
        let scene = Scene::new(800, 600);
        assert_eq!(scene.renderer.width, 800);
        assert_eq!(scene.renderer.height, 600);
    }

    #[test]
    fn test_color() {
        let color = Color::rgb(255, 0, 0);
        assert_eq!(color.to_hex(), "#FF0000");
    }

    #[test]
    fn test_vec3() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_geometry_builder() {
        let geom = GeometryBuilder::new().box_(1.0, 2.0, 3.0).build();

        match geom {
            Geometry::Box {
                width,
                height,
                depth,
            } => {
                assert_eq!(width, 1.0);
                assert_eq!(height, 2.0);
                assert_eq!(depth, 3.0);
            }
            _ => panic!("Expected box geometry"),
        }
    }

    #[test]
    fn test_preset_scenes() {
        let scene = PresetScenes::rotating_cube();
        assert_eq!(scene.objects.len(), 1);
        assert_eq!(scene.lights.len(), 2);
    }
}
