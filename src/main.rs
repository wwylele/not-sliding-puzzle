mod board;
mod entry;
mod res;

use winit::{event_loop::EventLoop, window::WindowBuilder};

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    let builder = WindowBuilder::new().with_title("Not Sliding Puzzle");
    let window = builder.build(&event_loop).unwrap();
    env_logger::init();
    pollster::block_on(entry::run(event_loop, window));
}
