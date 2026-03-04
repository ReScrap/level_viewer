use bevy::{
    pbr::MaterialExtension, prelude::*, render::render_resource::AsBindGroup, shader::ShaderRef,
};

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
pub(crate) struct TestMaterial {}

impl MaterialExtension for TestMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://level_viewer/shaders/test.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "embedded://level_viewer/shaders/test.wgsl".into()
    }
}
