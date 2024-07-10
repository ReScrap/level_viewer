use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension, OpaqueRendererMethod},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderRef},
};

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct Hologram {}

impl MaterialExtension for Hologram {
    fn fragment_shader() -> ShaderRef {
        "shaders/hologram.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/hologram.wgsl".into()
    }
}
