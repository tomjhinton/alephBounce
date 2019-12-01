const CANNON = require('cannon')
const THREE = require('three')
import './debug.js'
import Typed from 'typed.js';
import dat from 'dat.gui'
import Stats from 'stats.js'
import * as posenet from '@tensorflow-models/posenet'
import './style.scss'
import { drawKeypoints, drawSkeleton, drawHeatMapValues } from './demo_util'
const videoWidth = 1250
const videoHeight = 500
const stats = new Stats()
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'


//posenet
function isAndroid() {
  return /Android/i.test(navigator.userAgent)
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent)
}

function isMobile() {
  return isAndroid() || isiOS()
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw 'Browser API navigator.mediaDevices.getUserMedia not available'
  }

  const video = document.getElementById('video')
  video.width = videoWidth
  video.height = videoHeight

  const mobile = isMobile()
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined: videoHeight}
  })
  video.srcObject = stream

  return new Promise(resolve => {
    video.onloadedmetadata = () => {
      resolve(video)
    }
  })
}

async function loadVideo() {
  const video = await setupCamera()
  video.play()

  return video
}

const guiState = {
  algorithm: 'single-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '1.01',
    outputStride: 16,
    imageScaleFactor: 0.5
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5
  },
  multiPoseDetection: {
    maxPoseDetections: 2,
    minPoseConfidence: 0.1,
    minPartConfidence: 0.3,
    nmsRadius: 20.0
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true
  },
  net: null
}

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId
  }

  const cameraOptions = cameras.reduce((result, { label, deviceId }) => {
    result[label] = deviceId
    return result
  }, {})

  const gui = new dat.GUI({ width: 300, autoPlace: false  })

  // The single-pose algorithm is faster and simpler but requires only one person to be
  // in the frame or results will be innaccurate. Multi-pose works for more than 1 person
  const algorithmController = gui.add(
    guiState, 'algorithm', ['single-pose', 'multi-pose'])

  // The input parameters have the most effect on accuracy and speed of the network
  let input = gui.addFolder('Input')
  // Architecture: there are a few PoseNet models varying in size and accuracy. 1.01
  // is the largest, but will be the slowest. 0.50 is the fastest, but least accurate.
  const architectureController =
    input.add(guiState.input, 'mobileNetArchitecture', ['1.01', '1.00', '0.75', '0.50'])
  // Output stride:  Internally, this parameter affects the height and width of the layers
  // in the neural network. The lower the value of the output stride the higher the accuracy
  // but slower the speed, the higher the value the faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  //input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection')
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);
  //single.open();

  let multi = gui.addFolder('Multi Pose Detection')
  multi.add(
    guiState.multiPoseDetection, 'maxPoseDetections').min(1).max(20).step(1)
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0)
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0)
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0)

  let output = gui.addFolder('Output')
  output.add(guiState.output, 'showVideo')
  output.add(guiState.output, 'showSkeleton')
  output.add(guiState.output, 'showPoints')
  //output.open();


  architectureController.onChange(function (architecture) {
    guiState.changeToArchitecture = architecture
  })

  algorithmController.onChange(function (value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close()
        single.open()
        break
      case 'multi-pose':
        single.close()
        multi.open()
        break
    }
  })
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  //stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  //document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic happens.
 * This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output')
  const ctx = canvas.getContext('2d')
  const flipPoseHorizontal = true // since images are being fed from a webcam

  canvas.width = videoWidth
  canvas.height = videoHeight

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose()

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01 version
      guiState.net = await posenet.load(Number(guiState.changeToArchitecture));

      guiState.changeToArchitecture = null
    }

    // Begin monitoring code for frames per second
    stats.begin()

    // Scale an image down to a certain factor. Too large of an image will slow down
    // the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor
    const outputStride = Number(guiState.input.outputStride)

    let poses = []
    // console.log(poses)
    // console.log(poses.keypoints)




    let minPoseConfidence
    let minPartConfidence
    switch (guiState.algorithm) {
      case 'single-pose':
      const pose = await guiState.net.estimatePoses(video, {
        flipHorizontal: flipPoseHorizontal,
        decodingMethod: 'single-person'
      });

        poses.push(pose);
        // console.log(pose.keypoints)
        //color = `rgba(${pose.keypoints[9].position.x/100},${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},1)`

        //left square
        // console.log(poses.length)
        // console.log(poses)
        if(poses.length>= 1){
        if(poses[0][0].keypoints[9].position.x > 600){
          //synthA.triggerAttackRelease((poses[0][0].keypoints[9].position.y/100 )* poses[0][0].keypoints[9].position.x,0.01)

          // console.log(body.angularVelocity)
          body.position.x+=0.4
          console.log('right')
          // body.position.x+=0.1
        }

        if(poses[0][0].keypoints[9].position.x < 600){
          //synthA.triggerAttackRelease((poses[0][0].keypoints[9].position.y/100 )* poses[0][0].keypoints[9].position.x,0.01)

          // console.log(body.position)
          body.position.x-=0.4
          console.log('left')

            // body.position.x-=0.1
        }

        if(poses[0][0].keypoints[10].position.y < 200){
          //synthA.triggerAttackRelease((poses[0][0].keypoints[9].position.y/100 )* poses[0][0].keypoints[9].position.x,0.01)

          // console.log(body.velocity)
          //body.velocity.x-=0.01
          //console.log('left')

            body.position.z-=0.4
        }

        if(poses[0][0].keypoints[10].position.y > 200){
          //synthA.triggerAttackRelease((poses[0][0].keypoints[9].position.y/100 )* poses[0][0].keypoints[9].position.x,0.01)

          // console.log(body.velocity)
          //body.velocity.x-=0.01
          //console.log('left')

            body.position.z+=0.4
        }




        }






        minPoseConfidence = Number(
          guiState.singlePoseDetection.minPoseConfidence)
        minPartConfidence = Number(
          guiState.singlePoseDetection.minPartConfidence)
        break
      case 'multi-pose':
        poses = await guiState.net.estimateMultiplePoses(video, imageScaleFactor, flipPoseHorizontal, outputStride,
          guiState.multiPoseDetection.maxPoseDetections,
          guiState.multiPoseDetection.minPartConfidence,
          guiState.multiPoseDetection.nmsRadius)

        minPoseConfidence = Number(guiState.multiPoseDetection.minPoseConfidence)
        minPartConfidence = Number(guiState.multiPoseDetection.minPartConfidence)
        break
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight)


    if (guiState.output.showVideo) {
      ctx.save()
      ctx.scale(-1, 1)
      ctx.translate(-videoWidth, 0)
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight)
      ctx.restore()
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses[0].forEach(({ score, keypoints }) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx)
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx)
        }
      }
    })

    // End monitoring code for frames per second
    stats.end()


    requestAnimationFrame(poseDetectionFrame)
  }

  poseDetectionFrame()
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading available
 * camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights for version 1.01
  const net = await posenet.load()

  // document.getElementById('loading').style.display = 'none';
  // document.getElementById('main').style.display = 'block';

  let video

  try {
    video = await loadVideo();
  } catch(e) {
    let info = document.getElementById('info')
    info.textContent = 'this browser does not support video capture, or this device does not have a camera'
    info.style.display = 'block'
    throw e
  }

  setupGui([], net)
  setupFPS()
  detectPoseInRealTime(video, net)

}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia ||
  navigator.mozGetUserMedia
bindPage() /// kick off the demo



//TONE

//CANNNON && THREE
// Create a
var world, mass, body, shape, timeStep=1/60,
camera, scene, renderer, geometry, material, mesh, groundBody, floor, groundShape, physicsMaterial, ballShape, ballBody, radius, balls=[], ballMeshes=[], group, controls, ceilingBody, ceilingShape , leftWallShape, leftWallBody, rightWallShape, rightWallBody, frontWallShape, frontWallBody, backWallShape, backWallBody
initThree()
initCannon()
animate()

function initCannon() {
  world = new CANNON.World()
  world.gravity.set(0,-20,0)
  world.broadphase = new CANNON.NaiveBroadphase()
  world.solver.iterations = 10

  physicsMaterial = new CANNON.Material('slipperyMaterial')
  var physicsContactMaterial = new CANNON.ContactMaterial(physicsMaterial,physicsMaterial)
  physicsContactMaterial.friction = 0
  physicsContactMaterial.restitution = 0.9

  console.log(physicsContactMaterial)
  world.addContactMaterial(physicsContactMaterial)
  shape = new CANNON.Box(new CANNON.Vec3(1,1,1))

  const gameBox = new CANNON.Box(new CANNON.Vec3(10,20,10))
  const gameBody = new CANNON.Body({
    mass: 100, material: physicsMaterial
  })
  mass = 100
  body = new CANNON.Body({
    mass: 1, material: physicsMaterial
  })
  gameBody.addShape(gameBox)
  body.addShape(shape)
  body.angularVelocity.set(0,0,0)
  body.angularDamping = 0.2
  world.addBody(body)
  body.position.y = 0


  groundShape = new CANNON.Box(new CANNON.Vec3(30,30,10))
  groundBody = new CANNON.Body({ mass: 0, material: physicsMaterial })
  groundBody.addShape(groundShape)
  groundBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1,0,0),-Math.PI/2)
  groundBody.position.set(0,0,0)
  groundBody.position.y = -20
  world.addBody(groundBody)

  ceilingShape = new CANNON.Box(new CANNON.Vec3(30,30,10))
  ceilingBody = new CANNON.Body({ mass: 0, material: physicsMaterial })
  ceilingBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1,0,0),-Math.PI/2)
  ceilingBody.addShape(ceilingShape)
  ceilingBody.position.set(0,0,0)
  ceilingBody.position.y = 20
  world.addBody(ceilingBody)

  leftWallShape = new CANNON.Box(new CANNON.Vec3(20,10,20))
  leftWallBody = new CANNON.Body({ mass: 0, material: physicsMaterial })
  leftWallBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1,0,0),-Math.PI/2)
  leftWallBody.addShape(leftWallShape)
  leftWallBody.position.set(0,0,0)
  leftWallBody.position.z = -20
  world.addBody(leftWallBody)

  rightWallShape = new CANNON.Box(new CANNON.Vec3(20,10,20))
  rightWallBody = new CANNON.Body({ mass: 0, material: physicsMaterial })
  rightWallBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1,0,0),-Math.PI/2)
  rightWallBody.addShape(rightWallShape)
  rightWallBody.position.set(0,0,0)
  rightWallBody.position.z = 20
  world.addBody(rightWallBody)

  frontWallShape = new CANNON.Box(new CANNON.Vec3(20,10,20))
  frontWallBody = new CANNON.Body({ mass: 0, material: physicsMaterial })
  frontWallBody.addShape(frontWallShape)
  frontWallBody.position.set(0,0,0)
  frontWallBody.position.x = -40
  world.addBody(frontWallBody)

  backWallShape = new CANNON.Box(new CANNON.Vec3(20,10,20))
  backWallBody = new CANNON.Body({ mass: 0, material: physicsMaterial })
  backWallBody.addShape(backWallShape)
  backWallBody.position.set(0,0,0)
  backWallBody.position.x = 40
  world.addBody(backWallBody)



  console.log(ballBody)




  mass = 5, radius = 1


  ballShape = new CANNON.Sphere(radius)
  ballBody = new CANNON.Body({ mass: mass })
  ballBody.addShape(ballShape)
  ballBody.position.set(0,5,0)
  ballBody.linearDamping = 0.9
  world.addBody(ballBody)
  balls.push(ballBody)
  console.log(ballBody)
  console.log(world)
  //world.add(groundBody)
}
function initThree() {
  scene = new THREE.Scene()
  //camera
  camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 1, 100 )
  camera.position.z = 23
  camera.position.y = -3
  scene.add( camera )
  //lighting
  var Alight = new THREE.AmbientLight( 0x404040 ) // soft white light
  scene.add( Alight )
  const light = new THREE.DirectionalLight( 0xffffff )
  light.position.set( 40, 25, 10 )
  light.castShadow = true
  scene.add(light)


  //Objects
  geometry = new THREE.BoxGeometry( 2, 2, 2 )
  material =  new THREE.MeshPhongMaterial( { color: `rgba(${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},1)`, specular: `rgba(${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},1)` , shininess: 100, side: THREE.DoubleSide, opacity: 0.8,
    transparent: false } )


  var ballGeometry = new THREE.SphereGeometry(1, 32, 32)
  var ballMesh = new THREE.Mesh( ballGeometry, material )
  scene.add(ballMesh)
  ballMeshes.push(ballMesh)
  mesh = new THREE.Mesh( geometry, material )



  //BOX
  const materialWall = new THREE.MeshPhongMaterial( { color: `rgba(${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},1)`, specular: `rgba(${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},${Math.floor(Math.random()*255)},1)` , shininess: 100, side: THREE.DoubleSide, opacity: 0.8,
    transparent: true } )

  group = new THREE.Group();
  group.scale.set(4, 2, 2);


  setPlane("y",  Math.PI * 0.5, 0xff0000); //px
  setPlane("y", -Math.PI * 0.5, 0xff0000); //nx
  setPlane("x",  -Math.PI * 0.5, 0x00ff00); //ny
  setPlane("y",  0, 0x0000ff); //pz
  setPlane("y",  Math.PI, 0x0000ff);// nz

  function setPlane(axis, angle, color) {
    const planeGeom = new THREE.PlaneGeometry(10, 10, 10, 10)
    planeGeom.translate(0, 0, 5)
    switch (axis) {
      case 'y':
        planeGeom.rotateY(angle)
        break
      default:
        planeGeom.rotateX(angle)
    }
    const plane = new THREE.Mesh(planeGeom, new THREE.MeshBasicMaterial({color: color, side: THREE.DoubleSide}))

    group.add(plane)
  }
  group.quaternion.setFromAxisAngle(new CANNON.Vec3(1,0,0),-Math.PI/2)
  scene.add( mesh, floor, group )
  renderer = new THREE.WebGLRenderer()
  renderer.setSize( window.innerWidth, window.innerHeight )
  document.body.appendChild( renderer.domElement )
  controls = new OrbitControls( camera, renderer.domElement )
}





const cannonDebugRenderer = new THREE.CannonDebugRenderer( scene, world )




function animate() {
  //group.rotation.y +=0.01
  if(cannonDebugRenderer){
    cannonDebugRenderer.update()
  }
  controls.update()
  requestAnimationFrame( animate )
  updatePhysics()
  render()

}
function updatePhysics() {
  // Step the physics world
  world.step(timeStep)


  //console.log(mesh.position)
  // Copy coordinates from Cannon.js to Three.js
  mesh.position.copy(body.position)
  mesh.quaternion.copy(body.quaternion)
  // floor.quaternion.copy(groundBody.quaternion)
  // floor.quaternion.copy(groundBody.quaternion)
  for(var i=0; i<balls.length; i++){
    ballMeshes[i].position.copy(balls[i].position)
    ballMeshes[i].quaternion.copy(balls[i].quaternion)
  }
}
function render() {
  renderer.render( scene, camera )
}
