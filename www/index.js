import * as wasm from "notslidingpuzzle";

window.addEventListener('resize', resizeCanvas, false);

resizeCanvas();

function resizeCanvas() {
  var htmlCanvas = document.getElementById('canvas');
  htmlCanvas.width = window.innerWidth;
  htmlCanvas.height = window.innerHeight;
}


wasm.program_main();
