mod board;
mod entry;
mod res;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn program_main() {
    use wasm_bindgen::JsCast;
    use wgpu::web_sys;
    use winit::platform::web::WindowBuilderExtWebSys;
    use winit::{event_loop::EventLoop, window::WindowBuilder};

    let event_loop = EventLoop::new().unwrap();
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let builder = WindowBuilder::new().with_canvas(Some(canvas));
    let window = builder.build(&event_loop).unwrap();
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    wasm_bindgen_futures::spawn_local(entry::run(event_loop, window));
}

pub fn dummy() {
    let _ = &entry::run;
}
