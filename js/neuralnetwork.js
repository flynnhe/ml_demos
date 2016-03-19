// START OF NEURALNETWORK
function NeuralNetwork(lrRate, numHiddenLayers, numUnitsPerLayer, activateFn) {
  this.numHiddenLayers = numHiddenLayers;
  this.numUnitsPerLayer = numUnitsPerLayer;
  this.activateFn = activateFn;
  this.lrRate = lrRate;
  this.momentum = 0.9;
  this.weightDecay = 0.001;
  this.currEpoch = 1;
  this.weights = new Array(numHiddenLayers + 1);
  this.avgError = 0.0;
  numInputUnits = 4;
  numOutputUnits = 1;

  // set up the layers (currently only supporting same number of neurons in each hidden layer)
  var nodeCounts = new Array();
  nodeCounts.push(numInputUnits);
  for (var i = 0; i < numHiddenLayers; i++) {
    nodeCounts.push(numUnitsPerLayer);
  }
  nodeCounts.push(numOutputUnits);

  // set up the arrays to store the data during forward propagation
  outputs = new Array(numHiddenLayers + 2);
  tmp_outputs = new Array(numHiddenLayers + 2);

  // set up the arrays to store the deltas during back propagation
  deltasData = new Array(numHiddenLayers + 2);
  deltasW = new Array(numHiddenLayers + 1);

  // initialize the weights randomly
  for (var k = 0; k < numHiddenLayers + 1; k++) {
    var nFrom = nodeCounts[k] + 1; // +1 for bias
    var nTo = nodeCounts[k + 1] + 1;
    this.weights[k] = new Array(nFrom);
    deltasW[k] = new Array(nFrom);
    for (var i = 0; i < nFrom; i++) {
      this.weights[k][i] = new Array();
      deltasW[k][i] = new Array();
      for (var j = 0; j < nTo; j++) {
        if (j == 0) {
          this.weights[k][i].push(0); // don't have a weight going from bias to bias
        }
        else {
          this.weights[k][i].push(Math.random() * 2 - 1);
        }
        deltasW[k][i].push(0);
      }
    }
  }

  // set the base learning rate
  this.setLrRate = function(r) {
    this.lrRate = r;
  }

  // get the network's prediction for a given x
  this.predict = function(x) {
    tmp_outputs[0] = [1, x, x*x, x*x*x, x*x*x*x]; // add 1 for bias
    var data = forwardPropagate(x, this.weights, tmp_outputs, x, this.activateFn);
    return tmp_outputs[numHiddenLayers + 1][1];
  }

  // train the network to minimize the L2 Euclidean loss
  this.trainOneEpoch = function(examples) {
    var numExamples = examples.length;
    var indices = new Array(examples.length);
    for (var i = 0; i < examples.length; i++) {
      indices[i] = i;
    }
    // do forward pass through whole training set
    // randomize the training examples
    var order = shuffle(indices);
    this.avgError = 0.0;
    for (var idx = 0; idx < numExamples; idx++) {
      var n = order[idx];
      input = examples[n].x;
      target = examples[n].y; 
      outputs[0] = [1, input, input*input, input*input*input, input*input*input*input];
      var data = forwardPropagate(input, this.weights, outputs, target, this.activateFn);
      deltasData[numHiddenLayers + 1] = data[1];

      // compute the sensitivities (data derivs) for each layer except the last
      for (var k = numHiddenLayers; k >= 0; k--) {
        var nFrom = nodeCounts[k] + 1; // +1 for bias
        var nTo = nodeCounts[k + 1] + 1;
        deltasData[k] = new Array(nFrom);
        deltasData[k][0] = 0;
        for (var i = 1; i < nFrom; i++) {
          deltasData[k][i] = 0;
          for (var j = 1; j < nTo; j++) {
            deltasData[k][i] += this.weights[k][i][j] * deltasData[k+1][j];
          }
          if (k != 0) { // input layer is not activated
            deltasData[k][i] *= dActivate(outputs[k][i], this.activateFn);
          }
        }
      }

      // compute the weight updates for each layer
      for (var k = 0; k < numHiddenLayers + 1; k++) {
        var nFrom = nodeCounts[k] + 1; // +1 for bias
        var nTo = nodeCounts[k + 1] + 1;
        for (var j = 1; j < nTo; j++) {
          // bias updates
          deltasW[k][0][j] = (this.lrRate / Math.pow(2, k)) * deltasData[k+1][j] + this.momentum * deltasW[k][0][j];
          this.weights[k][0][j] -= deltasW[k][0][j];
          for (var i = 1; i < nFrom; i++) {
            var dEdW = 0;
            if (k != 0) {
              dEdW = activate(outputs[k][i], this.activateFn) * deltasData[k+1][j];
              dEdW += this.weightDecay * this.weights[k][i][j];
            }
            else {
              dEdW = outputs[k][i] * deltasData[k+1][j];
              dEdW += this.weightDecay * this.weights[k][i][j];
            }
            deltasW[k][i][j] = (this.lrRate / Math.pow(2, k)) * dEdW + this.momentum * deltasW[k][i][j];
            this.weights[k][i][j] -= deltasW[k][i][j];
          }
        }
      }
      this.avgError += (this.predict(examples[n].x) - target) * (this.predict(examples[n].x) - target);
    }
    this.avgError /= numExamples;
    this.currEpoch++;
  }

}
// END OF NEURALNETWORK

// normalize points to have zero mean and unit variance
function normalize(points) {
  var n = points.length;
  var mean = [0, 0, 0, 0, 0];
  var std = [0, 0, 0, 0, 0];
  for (var i = 0; i < n; i++) { // compute mean
    mean[0] += points[i].x;
    mean[1] += points[i].x2;
    mean[2] += points[i].x3;
    mean[3] += points[i].x4;
    mean[4] += points[i].y;
  }
  mean = [mean[0]/n, mean[1]/n, mean[2]/n, mean[3]/n, mean[4]/n];

  for (var i = 0; i < n; i++) { // compute std
    std[0] += (points[i].x - mean[0]) * (points[i].x - mean[0]);
    std[1] += (points[i].x2 - mean[1]) * (points[i].x2 - mean[1]);
    std[2] += (points[i].x3 - mean[2]) * (points[i].x3 - mean[2]);
    std[3] += (points[i].x4 - mean[3]) * (points[i].x4 - mean[3]);
    std[4] += (points[i].y - mean[4]) * (points[i].y - mean[4]);  
  }
  std = [Math.sqrt(std[0]/n),Math.sqrt(std[1]/n),Math.sqrt(std[2]/n),Math.sqrt(std[3]/n),Math.sqrt(std[4]/n)];

  // subtract mean and divide by std
  for (var i = 0; i < n; i++) {
    points[i].x = (points[i].x - mean[0]) / std[0];
    points[i].x2 = (points[i].x2 - mean[1]) / std[1];
    points[i].x3 = (points[i].x3 - mean[2]) / std[2];
    points[i].x4 = (points[i].x4 - mean[3]) / std[3];
    points[i].y = (points[i].y - mean[4]) / std[4];
  }

  return points;
}

function forwardPropagate(inputs, weights, outputs, targets, activateFn)
{
  var numLayers = weights.length;
  var numHiddenLayers = numLayers - 1;
  var sum = 0;

  for (var i = 1; i <= numLayers; i++) {
    if (i == 1) {
      sum = computeSum(weights[i-1], [1, inputs, inputs*inputs, inputs*inputs*inputs, inputs*inputs*inputs*inputs]);
    }
    else {
      sum = computeSum(weights[i-1], activate(outputs[i-1], activateFn));
    }
    outputs[i] = sum;
  }

  var errorData = computeError(targets, outputs[numHiddenLayers + 1], weights);
  return errorData;
}

function computeSum(weights, inputs) {
  var nFrom = weights.length;
  var nTo = weights[0].length;
  var sum = new Array(nTo);
  sum[0] = 0; // dummy bias
  for (var j = 1; j < nTo; j++) {
    sum[j] = weights[0][j]; // bias
    for (var i = 1; i < nFrom; i++) {
      sum[j] += weights[i][j] * inputs[i];
    }
  }
  return sum;
}

function computeError(target, output, weights) {
  target1 = [output[0], target]; // add term to match bias
  var numUnits = target1.length;
  var deltasO = new Array(numUnits);
  var error = 0;
  deltasO[0] = 0;
  for (var j = 1; j < numUnits; j++) {
    error += dActivate(output[j], activateFn) * (output[j] - target1[j]) * (output[j] - target1[j]);
    error += l2norm(weights);
    deltasO[j] = output[j] - target1[j];
  }
  error *= 0.5;
  return [error, deltasO];
}

function l2norm(weights) {
  var norm = 0.0;
  for (var i = 0; i < weights.length; i++) {
    var nFrom = weights[i].length;
    var nTo = weights[i][0].length; 
    for (var j = 0; j < nFrom; j++) {
			for (var k = 0; k < nTo; k++) {
				norm += weights[i][j][k] * weights[i][j][k];
			}
    }
  }
  return Math.sqrt(norm);
}

function activate(x, activateFn) {
  if (typeof x == 'number') x = [x];
  switch (activateFn) {
    case 'sigmoid':
      return sigmoid(x);
    case 'tanh':
      return tanh(x);
    case 'relu':
      return relu(x);
    default:
      return identity(x);
  }
}

function dActivate(y, activateFn) {
  if (typeof y == 'number') y = [y];
  switch (activateFn) {
    case 'sigmoid':
      return dSigmoid(y);
    case 'tanh':
      return dTanh(y);
    case 'relu':
      return dRelu(y);
    default:
      return dIdentity(y);
  }
}

function identity(x) {
  if (typeof x == 'number') x = [x];
  return x;
}

function sigmoid(x) {
  if (typeof x == 'number') x = [x];
  var y = new Array(x.length);
  for (var i = 0; i < x.length; i++) {
    y[i] = 1 / (1 + Math.exp(-x[i]));
  }
  return y;
}

function tanh(x) {
  if (typeof x == 'number') x = [x];
  var y = new Array(x.length);
  for (var i = 0; i < x.length; i++) {
    y[i] = (Math.exp(x[i]) - Math.exp(-x[i])) / (Math.exp(x[i]) + Math.exp(-x[i]));
  }
  return y;
}

function relu(x) {
  if (typeof x == 'number') x = [x];
  var y = new Array(x.length);
  for (var i = 0; i < x.length; i++) {
    y[i] = Math.max(0, x[i]);
  }
  return y;
}

function dIdentity(y) {
  if (typeof y == 'number') y = [y];
  var dy = new Array(y.length);
  for (var i = 0; i < y.length; i++) {
    dy[i] = 1;
  }
  return dy;
}

function dSigmoid(y) {
  if (typeof y == 'number') y = [y];
  var dy = new Array(y.length);
  for (var i = 0; i < y.length; i++) {
    dy[i] = sigmoid(y[i]) * (1 - sigmoid(y[i]));
  }
  return dy;
}

function dTanh(y) {
  if (typeof y == 'number') y = [y];
  var dy = new Array(y.length);
  for (var i = 0; i < y.length; i++) {
    dy[i] = 1 - (tanh(y[i]))*(tanh(y[i]));
  }
  return dy;
}

function dRelu(y) {
  if (typeof y == 'number') y = [y];
  var dy = new Array(y.length);
  for (var i = 0; i < y.length; i++) {
    if (y[i] > 0) {
        dy[i] = 1;
    } else {
        dy[i] = 0;
    }
  }
  return dy;
}

function shuffle(array) {
  var copy = [], n = array.length, i;

  // While there remain elements to shuffle…
  while (n) {

    // Pick a remaining element…
    i = Math.floor(Math.random() * array.length);

    // If not already shuffled, move it to the new array.
    if (i in array) {
      copy.push(array[i]);
      delete array[i];
      n--;
    }
  }

  return copy;
}
