// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { initializeApp } from 'https://www.gstatic.com/firebasejs/11.1.0/firebase-app.js'

// If you enabled Analytics in your project, add the Firebase SDK for Google Analytics
import { getAnalytics } from 'https://www.gstatic.com/firebasejs/11.1.0/firebase-analytics.js'

// Add Firebase products that you want to use
import { getAuth } from 'https://www.gstatic.com/firebasejs/11.1.0/firebase-auth.js'
import { getFirestore } from 'https://www.gstatic.com/firebasejs/11.1.0/firebase-firestore.js'

import { getDatabase, ref, set, update, remove } from 'https://www.gstatic.com/firebasejs/11.1.0/firebase-database.js';

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js"

const firebaseConfig = {
  apiKey: "AIzaSyCdfvtPi52SI5RgohLJ1Fkb7jqUF5_O77U",
  authDomain: "simplegaze-34f27.firebaseapp.com",
  databaseURL: "https://simplegaze-34f27-default-rtdb.firebaseio.com",
  projectId: "simplegaze-34f27",
  storageBucket: "simplegaze-34f27.firebasestorage.app",
  messagingSenderId: "425579519055",
  appId: "1:425579519055:web:8d192926ca4690feedde3a"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);


import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3"
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision
const canvasElement = document.getElementById("output_canvas")
const udElement = document.getElementById("mesh_canvas")
const refTextbox = document.getElementById("refTextbox");
const imageBlendShapes = document.getElementById("image-blend-shapes")
const videoBlendShapes = document.getElementById("video-blend-shapes")

let faceLandmarker
let runningMode = "IMAGE"
let enableWebcamButton
let webcamRunning = false
const videoWidth = 480

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
let currentRef = 'Name';

refTextbox.addEventListener("input", () => {
  const newRef = refTextbox.value.trim();
  if (newRef && newRef !== currentRef) {
    const oldRef = ref(database, currentRef);
    remove(oldRef); // Remove the previous ref
    currentRef = newRef;
  }
});

async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  )
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1
  })
  canvasElement.classList.remove("invisible")
  udElement.classList.remove("invisible")
  document.getElementById('pause-play-button').src = "https://cdn-icons-png.flaticon.com/512/62/62070.png";
}
enableCam()
createFaceLandmarker()

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick")

// Now let's go through all of these and add a click event listener.
for (let imageContainer of imageContainers) {
  // Add event listener to the child element whichis the img element.
  imageContainer.children[0].addEventListener("click", handleClick)
}

function drawPixel(ctx, x, y, size, color) {
  ctx.fillStyle = color;
  ctx.fillRect(Math.floor(x - size / 2), Math.floor(y - size / 2), size, size);
}

function drawPixelatedLandmarks(landmarks, ctx, size, color) {
  for (const landmark of landmarks) {
    drawPixel(ctx, landmark.x * canvasElement.width, landmark.y * canvasElement.height, size, color);
  }
}

function getBoundingBox(landmarks, indices) {
  let minX = Number.MAX_VALUE, minY = Number.MAX_VALUE;
  let maxX = Number.MIN_VALUE, maxY = Number.MIN_VALUE;
  for (const idx of indices) {
    const x = landmarks[idx].x;
    const y = landmarks[idx].y;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function getBoundingBoxFromLines(landmarks, indices) {
  let minX = Number.MAX_VALUE, minY = Number.MAX_VALUE;
  let maxX = Number.MIN_VALUE, maxY = Number.MIN_VALUE;
  for (const idx of indices) {
    const x = landmarks[idx.start].x;
    const y = landmarks[idx.start].y;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function drawMinecraftLandmarks(cast, blendShapes, ctx, width, height) {
  const landmarks = cast.landmarks;
  const bs = {};
  blendShapes[0].categories.map(shape => {
    bs[shape.displayName || shape.categoryName] = +shape.score;
  });
  const leftEyeBB = getBoundingBoxFromLines(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
  const rightEyeBB = getBoundingBoxFromLines(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
  const lebb = getBoundingBoxFromLines(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW);
  const rebb = getBoundingBoxFromLines(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW);
  const leftEyebrowBB = {x: leftEyeBB.x, y: lebb.y + lebb.h / 2, w: leftEyeBB.w, h: 0.02};
  const rightEyebrowBB = {x: rightEyeBB.x, y: rebb.y + rebb.h / 2, w: rightEyeBB.w, h: 0.02};

  const percentTurnedX = Math.max(Math.min(cast.rotation.yaw / 200, 0.5), -0.5) + 0.5;
  const irisMovedLeft = percentTurnedX + 0.2*(bs.eyeLookInLeft - bs.eyeLookOutLeft);
  const irisMovedRight = percentTurnedX + 0.2*(bs.eyeLookOutRight - bs.eyeLookInRight);

  const irisWidth = 0.05;
  const leftIrisHeight = Math.min(irisWidth, leftEyeBB.h * 1.1);
  const leftIrisCenterX = leftEyeBB.x + (leftEyeBB.w * irisMovedLeft) - irisWidth/2;
  const leftIrisCenterY = leftEyeBB.y + leftEyeBB.h/2 - leftIrisHeight/2;
  const rightIrisHeight = Math.min(irisWidth, rightEyeBB.h * 1.1);
  const rightIrisCenterX = rightEyeBB.x + (rightEyeBB.w * irisMovedRight) - irisWidth/2;
  const rightIrisCenterY = rightEyeBB.y + rightEyeBB.h/2 - rightIrisHeight/2;

  ctx.fillStyle = '#b6dcf0';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = 'white';
  ctx.fillRect(leftEyeBB.x * width, leftEyeBB.y * height, leftEyeBB.w * width, leftEyeBB.h * height);
  ctx.fillRect(rightEyeBB.x * width, rightEyeBB.y * height, rightEyeBB.w * width, rightEyeBB.h * height);
  ctx.fillStyle = 'brown';
  ctx.fillRect(leftEyebrowBB.x * width, leftEyebrowBB.y * height, leftEyebrowBB.w * width, leftEyebrowBB.h * height);
  ctx.fillRect(rightEyebrowBB.x * width, rightEyebrowBB.y * height, rightEyebrowBB.w * width, rightEyebrowBB.h * height);
  ctx.fillStyle = 'black';
  ctx.fillRect(leftIrisCenterX * width, leftIrisCenterY * height, irisWidth * width, leftIrisHeight * height);
  ctx.fillRect(rightIrisCenterX * width, rightIrisCenterY * height, irisWidth * width, rightIrisHeight * height);

  const mouthBB = getBoundingBox(landmarks, [80, 88, 82, 13, 312, 87, 14, 317, 310, 318]);
  const mouthBBL = getBoundingBox(landmarks, [80, 88, 78]);
  const mouthBBR = getBoundingBox(landmarks, [310, 318, 308]);

  if (bs.mouthSmileLeft > 0.5 || bs.mouthSmileRight > 0.5) {
    ctx.fillRect(mouthBB.x * width, mouthBB.y * height, mouthBB.w * width, mouthBB.h * height);
    ctx.fillRect(mouthBBL.x * width, mouthBB.y * height, mouthBBL.w * width, mouthBBL.h * height);
    ctx.fillRect(mouthBBR.x * width, mouthBB.y * height, mouthBBR.w * width, mouthBBR.h * height);

    postShapes({
      mouthLeft: {x: mouthBBL.x, y: mouthBB.y, w: mouthBBL.w, h: mouthBBL.h},
      mouthRight: {x: mouthBBR.x, y: mouthBB.y, w: mouthBBR.w, h: mouthBBR.h},
      mouthCenter: mouthBB,
      leftEyebrow: leftEyebrowBB,
      rightEyebrow: rightEyebrowBB,
      leftEye: leftEyeBB,
      rightEye: rightEyeBB,
      leftIris: {x: leftIrisCenterX, y: leftIrisCenterY, w: irisWidth, h: leftIrisHeight},
      rightIris: {x: rightIrisCenterX, y: rightIrisCenterY, w: irisWidth, h: rightIrisHeight},
    });
  } else {
    // const mouthBB = getBoundingBox(landmarks, [78, 82, 13, 312, 87, 14, 317, 308]);
    ctx.fillRect(mouthBB.x * width, mouthBB.y * height, mouthBB.w * width, mouthBB.h * height);
    ctx.fillRect(mouthBBL.x * width, mouthBB.y * height, mouthBBL.w * width, mouthBB.h * height);
    ctx.fillRect(mouthBBR.x * width, mouthBB.y * height, mouthBBR.w * width, mouthBB.h * height);

    postShapes({
      mouthLeft: {x: mouthBBL.x, y: mouthBB.y, w: mouthBBL.w, h: mouthBB.h},
      mouthRight: {x: mouthBBR.x, y: mouthBB.y, w: mouthBBR.w, h: mouthBB.h},
      mouthCenter: mouthBB,
      leftEyebrow: leftEyebrowBB,
      rightEyebrow: rightEyebrowBB,
      leftEye: leftEyeBB,
      rightEye: rightEyeBB,
      leftIris: {x: leftIrisCenterX, y: leftIrisCenterY, w: irisWidth, h: leftIrisHeight},
      rightIris: {x: rightIrisCenterX, y: rightIrisCenterY, w: irisWidth, h: rightIrisHeight},
    });
  }
}

let paused = false

function postShapes(res) {
  if (currentRef === 'Name' || paused) return;
  const dbRef = ref(database, currentRef);
  update(dbRef, res);
}

//0.47073494637625657 0.005389876663684845 0.2982533574104309
//0.2767046103075495 0.8386083245277405 0.0028104418888688087


// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
  if (!faceLandmarker) {
    console.log("Wait for faceLandmarker to load before clicking!")
    return
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE"
    await faceLandmarker.setOptions({ runningMode })
  }
  // Remove all landmarks drawed before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas")
  for (var i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i]
    n.parentNode.removeChild(n)
  }

  // We can call faceLandmarker.detect as many times as we like with
  // different image data each time. This returns a promise
  // which we wait to complete and then call a function to
  // print out the results of the prediction.
  const faceLandmarkerResult = faceLandmarker.detect(event.target)
  const canvas = document.createElement("canvas")
  canvas.setAttribute("class", "canvas")
  canvas.setAttribute("width", event.target.naturalWidth + "px")
  canvas.setAttribute("height", event.target.naturalHeight + "px")
  canvas.style.left = "0px"
  canvas.style.top = "0px"
  canvas.style.width = `${event.target.width}px`
  canvas.style.height = `${event.target.height}px`

  event.target.parentNode.appendChild(canvas)
  const ctx = canvas.getContext("2d")
  const drawingUtils = new DrawingUtils(ctx)
  for (const landmarks of faceLandmarkerResult.faceLandmarks) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#C0C0C070", lineWidth: 1 }
    )
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      { color: "#FF3030" }
    )
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      { color: "#FF3030" }
    )
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      { color: "#30FF30" }
    )
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      { color: "#30FF30" }
    )
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      { color: "#E0E0E0" }
    )
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: "#E0E0E0"
    })
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      { color: "#FF3030" }
    )
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
      { color: "#30FF30" }
    )
  }
  drawBlendShapes(imageBlendShapes, faceLandmarkerResult.faceBlendshapes)
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam")

const canvasCtx = canvasElement.getContext("2d")
const udCtx = udElement.getContext("2d")

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

// If webcam supported, add event listener to button for when user
// wants to activate it.
// if (hasGetUserMedia()) {
//   enableWebcamButton = document.getElementById("webcamButton")
//   enableWebcamButton.addEventListener("click", enableCam)
// } else {
//   console.warn("getUserMedia() is not supported by your browser")
// }

// Enable the live webcam view and start detection.
function enableCam(event) {
  webcamRunning = true;

  // getUsermedia parameters.
  const constraints = {
    video: true
  }

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(stream => {
    video.srcObject = stream
    video.addEventListener("loadeddata", predictWebcam)
  })
}

function undistortMesh(mesh, corners, dimensions) {
    const { width, height } = dimensions;
    const { topLeft, topRight, bottomLeft, bottomRight } = corners;

    // Helper function to perform bilinear interpolation
    function bilinearInterpolate(x, y) {
        const topX = (x - topLeft.x) / (topRight.x - topLeft.x);
        const leftY = (y - topLeft.y) / (bottomLeft.y - topLeft.y);
        const bottomX = (x - bottomLeft.x) / (bottomRight.x - bottomLeft.x);
        const rightY = (y - topRight.y) / (bottomRight.y - topRight.y);
        const finalX = (topX * (y - topLeft.y) + bottomX * (1 + topLeft.y - y)) / 2;
        const finalY = (leftY * (x - topLeft.x) + rightY * (1 + topLeft.x - x)) / 2;
        return { x: finalX, y: finalY + 0.5 };
    }

    // Map each point in the mesh to the flat surface
    const transformedMesh = mesh.map(({ x, y }) => {
        const normalizedX = x / width; // Map x to [0, 1]
        const normalizedY = y / height; // Map y to [0, 1]
        return bilinearInterpolate(normalizedX, normalizedY);
    });

    return transformedMesh;
}

function alignFaceToCamera(vertices, tipIndex, baseIndex, leftEyeIndex, rightEyeIndex) {
    // Convert vertices to Three.js Vector3 objects
    const vectorVertices = vertices.map(v => new THREE.Vector3(v.x, v.y, v.z));

    // Get the nose and eye points
    const tip = vectorVertices[tipIndex];
    const base = vectorVertices[baseIndex];
    const leftEye = vectorVertices[leftEyeIndex];
    const rightEye = vectorVertices[rightEyeIndex];

    // Compute the original distance between the tip and base of the nose
    const originalNoseLength = tip.distanceTo(base);

    // Compute the scaling factor to achieve the target nose length
    const targetNoseLength = 0.2;
    const scaleFactor = targetNoseLength / originalNoseLength;

    // Apply scaling to all vertices
    const scaledVertices = vectorVertices.map(vertex => vertex.clone().multiplyScalar(scaleFactor));

    // Get the scaled tip and base
    const scaledTip = scaledVertices[tipIndex];
    const scaledBase = scaledVertices[baseIndex];
    const scaledLeftEye = scaledVertices[leftEyeIndex];
    const scaledRightEye = scaledVertices[rightEyeIndex];

    // Compute the forward vector (nose direction)
    const upVector = new THREE.Vector3().subVectors(scaledTip, scaledBase).normalize();

    // Compute the up vector (eyes direction)
    const eyeMidpoint = new THREE.Vector3().addVectors(scaledLeftEye, scaledRightEye).multiplyScalar(0.5);
    const forwardVector = new THREE.Vector3().subVectors(eyeMidpoint, scaledBase).normalize();

    // Compute the right vector using the cross product
    const rightVector = new THREE.Vector3().crossVectors(forwardVector, upVector).normalize();

    // Recompute the true up vector to ensure orthogonality
    const correctedUpVector = new THREE.Vector3().crossVectors(rightVector, forwardVector).normalize();

    // Create a transformation matrix for aligning the face
    const matrix = new THREE.Matrix4();
    matrix.makeBasis(rightVector, correctedUpVector, forwardVector);

    // Extract the rotation in radians as Euler angles
    const euler = new THREE.Euler().setFromRotationMatrix(matrix);

    // Convert radians to degrees
    const pitch = THREE.MathUtils.radToDeg(euler.x); // Rotation around X-axis
    const yaw = THREE.MathUtils.radToDeg(euler.y);   // Rotation around Y-axis
    const roll = THREE.MathUtils.radToDeg(euler.z);  // Rotation around Z-axis

    // Invert the matrix to align the mesh
    const inverseMatrix = matrix.clone().invert();

    // Apply the alignment transformation to all scaled vertices
    let transformedVertices = scaledVertices.map(vertex => vertex.clone().applyMatrix4(inverseMatrix));

    // Get the transformed position of the nose tip
    const transformedTip = transformedVertices[tipIndex];

    // Compute the translation to move the tip of the nose to (0.5, 0.5)
    const translation = new THREE.Vector3(0.5, 0.5, 0).sub(transformedTip);

    // Apply the translation to all vertices
    transformedVertices = transformedVertices.map(vertex => vertex.add(translation));

    // Convert the transformed vertices back to the original format
    const finalVertices = transformedVertices.map(vertex => ({ x: vertex.x, y: vertex.y, z: vertex.z }));

    return {
        landmarks: finalVertices,
        rotation: {
            pitch, // X-axis rotation
            yaw,   // Y-axis rotation
            roll   // Z-axis rotation
        }
    };
}

let lastVideoTime = -1
let results = undefined
const drawingUtils = new DrawingUtils(canvasCtx)
async function predictWebcam() {
  const radio = video.videoHeight / video.videoWidth
  video.style.width = videoWidth + "px"
  video.style.height = videoWidth * radio + "px"
  canvasElement.style.width = videoWidth + "px"
  canvasElement.style.height = videoWidth * radio + "px"
  canvasElement.width = video.videoWidth
  canvasElement.height = video.videoHeight

  udElement.width = canvasElement.width;
  udElement.height = canvasElement.height;
  udElement.style.width = videoWidth + "px"
  udElement.style.height = videoWidth * radio + "px"
  // Now let's start detecting the stream.
  if (faceLandmarker && !paused) {
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO"
      await faceLandmarker.setOptions({ runningMode: runningMode })
    }
    let startTimeMs = performance.now()
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime
      results = faceLandmarker.detectForVideo(video, startTimeMs)
    }
  }
  
  if (results && results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
      drawPixelatedLandmarks(landmarks, canvasCtx, 3, "#C0C0C0");
      drawMinecraftLandmarks(alignFaceToCamera(landmarks, 1, 6, 226, 446), results.faceBlendshapes, udCtx, canvasElement.width, canvasElement.height);
      // drawPixelatedLandmarks(alignFaceToCamera(landmarks, 1, 6, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE[0].start, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE[0].start).landmarks, udCtx, 3, "#C0C0C0");
      
      // console.log(landmarks, alignNoseToCamera(landmarks, 1, 6));
      break;
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      //   { color: "#C0C0C070", lineWidth: 1 }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      //   { color: "#FF3030" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      //   { color: "#FF3030" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      //   { color: "#30FF30" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      //   { color: "#30FF30" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      //   { color: "#E0E0E0" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_LIPS,
      //   { color: "#F2CECE" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      //   { color: "#FF3030" }
      // )
      // drawingUtils.drawConnectors(
      //   landmarks,
      //   FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
      //   { color: "#30FF30" }
      // )
    }
  }
  // drawBlendShapes(videoBlendShapes, results.faceBlendshapes)
  // postBlendShapes(results.faceBlendshapes)



  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam)
  }
}

// function postBlendShapes(blendShapes) {
//   if (!blendShapes.length) {
//     return
//   }

//   const res = {};

//   blendShapes[0].categories.map(shape => {
//     const name = shape.displayName || shape.categoryName;
//     res[name] = +shape.score * 1;
//   })

//   const dbRef = ref(database, 'wyldevin');
//   update(dbRef, res);
// }

function drawBlendShapes(el, blendShapes) {
  if (!blendShapes.length) {
    return
  }

  // console.log(blendShapes[0])

  let htmlMaker = ""
  blendShapes[0].categories.map(shape => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName ||
          shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${+shape.score *
          100}% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `
  })

  el.innerHTML = htmlMaker
}


document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('pause-play-button');
    const video = document.getElementById('webcam');

    button.addEventListener('click', () => {
      if (!faceLandmarker) return;
      if (paused) {
        paused = false;
        button.src = "https://cdn-icons-png.flaticon.com/512/62/62070.png";
        button.alt = "Pause";
        canvasElement.classList.remove("invisible")
        udElement.classList.remove("invisible")
      } else {
        paused = true;
        button.src = "https://cdn-icons-png.flaticon.com/512/0/375.png";
        button.alt = "Play";
        canvasElement.classList.add("invisible")
        udElement.classList.add("invisible")
      }
    });
  });
