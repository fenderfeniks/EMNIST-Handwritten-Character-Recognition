// Variables for referencing the canvas and 2dcanvas context
var canvas, ctx;

// Variables to keep track of the mouse position and left-button status
var mouseX, mouseY, mouseDown = 0;

// Draws a dot at a specific position on the supplied canvas name
// Parameters are: A canvas context, the x position, the y position, the size of the dot
function drawDot(ctx, x, y) {
  let pxData = ctx.getImageData(x, y, 28, 28);
  pxData.data[0] = 0;
  pxData.data[1] = 0;
  pxData.data[2] = 0;
  pxData.data[3] = 255;
  ctx.putImageData(pxData, x, y);
}

// Clear the canvas context using the canvas width and height
function clearCanvas(canvas, ctx) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Keep track of the mouse button being pressed and draw a dot at current location
function sketchpad_mouseDown() {
  mouseDown = 1;
  drawDot(ctx, mouseX / 5, mouseY / 5);
}

// Keep track of the mouse button being released
function sketchpad_mouseUp() {
  mouseDown = 0;
}

// Keep track of the mouse position and draw a dot if mouse button is currently pressed
function sketchpad_mouseMove(e) {
  // Update the mouse co-ordinates when moved
  getMousePos(e);

  // Draw a dot if the mouse button is currently being pressed
  if (mouseDown == 1) {
    drawDot(ctx, mouseX / 5, mouseY / 5);
  }
}

function reset() {
  clearCanvas(canvas, ctx);
}

// Get the current mouse position relative to the top-left of the canvas
function getMousePos(e) {
  if (!e)
    var e = event;

  if (e.offsetX) {
    mouseX = e.offsetX;
    mouseY = e.offsetY;
  } else if (e.layerX) {
    mouseX = e.layerX;
    mouseY = e.layerY;
  }
}

// Set-up the canvas and add our event handlers after the page has loaded
function init() {
  // Get the specific canvas element from the HTML document
  canvas = document.getElementById('sketchpad');

  // If the browser supports the canvas tag, get the 2d drawing context for this canvas
  if (canvas.getContext) {
    ctx = canvas.getContext('2d');
    ctx.mozImageSmoothingEnabled = false;
    ctx.webkitImageSmoothingEnabled = false;
    ctx.msImageSmoothingEnabled = false;
    ctx.imageSmoothingEnabled = false;
  }

  // Check that we have a valid context to draw on/with before adding event handlers
  if (ctx) {
    canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
    canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
    window.addEventListener('mouseup', sketchpad_mouseUp, false);
  }
}

function predictData() {
  // const normalarray = Array.prototype.slice.call(imgdata);
  const data = ctx.getImageData(0,0,28,28);
  var img = []
  for(let i = 0; i < 28 * 28; i++) {
    img.push(data.data[4 * i + 3]);
  }
  $.ajax({
    url: "/api/predict",
    type: "POST",
    headers: {
      "Accept": "application/json"
    },
    data: JSON.stringify(img),
    processData: false,
    success: (res, status) => {
      console.log(res);
      document.getElementById("prediction").innerHTML = res.prediction;
    }
  });
}
