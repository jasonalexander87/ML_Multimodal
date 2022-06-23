

async function loadModel(){	
  	
    // loads the model
    try {
      
      const model = await tf.loadLayersModel('localStorage:/tensorflow/model.json'); 
      y = model.predict(tf.zeros([1]))
      document.getElementById("demo").innerHTML = y;  

  } catch(e) {
    document.getElementById("demo").innerHTML = e;  }
  }

  async function load(){

    try {
      const uploadJSONInput = document.getElementById('upload-json');
      const uploadWeightsInput = document.getElementById('upload-weights');
      const model = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
      y = model.predict(tf.zeros([1]))
      document.getElementById("demo").innerHTML = y;  

  } catch(e) {
    document.getElementById("demo").innerHTML = e;  }

  }

  async function predictModel(){
    
    // gets image data
    imageData = getData();
    
    // converts from a canvas data object to a tensor
    image = tf.browser.fromPixels(imageData)
    
    // pre-process image
    image = tf.image.resizeBilinear(image, [28,28]).sum(2).expandDims(0).expandDims(-1)
    
    // gets model prediction
    y = model.predict(image);
    
    // replaces the text in the result tag by the model prediction
    document.getElementById('result').innerHTML = "Prediction: " + y.argMax(1).dataSync();
  }