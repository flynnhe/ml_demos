// START OF RANDOMFOREST
function RandomForest(numTrees, maxDepth, numHypotheses) {
  this.numTrees = numTrees;
  this.maxDepth = maxDepth;
  this.numHypotheses = numHypotheses;
  this.trees = new Array();

  // initialise data
  for (var i = 0; i < this.numTrees; i++) {
    this.trees.push(new Object());
  }

  // generate a forest of decision trees
  this.train = function(examples) {
    for (var i = 0; i < this.numTrees; i++) {
      var curr_examples = this.getRandomSubset(examples, Math.floor(examples.length/2));
      this.trees[i] = new DecisionTree(this.maxDepth, this.numHypotheses);
      this.trees[i].train(0, curr_examples);
    }
  }

  // generate n random training examples from the full training set
  this.getRandomSubset = function(examples, n) {
    var arr = [];
    while (arr.length < n) {
      var r = Math.floor(Math.random() * examples.length);
      var found = false;
      for (var i = 0; i < arr.length; i++) {
        if (arr[i] == r) {
          found = true;
          break
        }
      }
      if (!found) {
        arr.push(examples[r]);
      }
    }
    return arr;
  }

  // predict the label of a new example
  this.predict = function(x, y) {
    var val = 0.0;
    for (var i = 0; i < this.numTrees; i++) {
      val += this.trees[i].predict(x, y);
    }
    val /= this.numTrees;
    return val;
  }

  // compute the misclassification error over all examples
  this.computeError = function(examples) {
    var N = examples.length;
    var misclassified = 0;
    for (var i = 0; i < N; i++) {
      var act = examples[i].label;
      var est = this.predict(examples[i].x, examples[i].y);
      if (this.sign(act) != this.sign(est)) {
        misclassified++;
      }
    }
    return misclassified / N;
  };

  this.sign = function(x) {
    return x < 0 ? -1 : 1;
  };
}
// END OF RANDOMFOREST

// START OF DECISIONTREE
function DecisionTree(maxDepth, numHypotheses) {
  this.maxDepth = maxDepth;
  this.numHypotheses = numHypotheses;
  this.size = Math.pow(2, maxDepth + 1) - 1;
  this.data = new Array();

  // initialise data
  for (var i = 0; i < this.size; i++) {
    this.data.push(new Object());
  }

  // grow a decision tree at node idx
  this.train = function(idx, examples) {

    // if we've reached the maximum tree depth, make this node a leaf
    if (this.isLeaf(idx)) {
      this.data[idx] = {
        isLeaf: true,
        threshold: [],
        label: this.getVote(examples)
      };
      return;
    }

    // if all the examples are the same, make this a leaf node
    if (this.allSame(examples)) {
      this.data[idx] = {
        isLeaf: true,
        threshold: [],
        label: this.getVote(examples)
      };
      return;
    }

    // generate a number of random hypotheses and take the one
    // having the highest information gain
    var maxInfoGain = [0, [],
      []
    ];
    var besta = 0,
      bestb = 0,
      bestc = 0;
    for (var i = 0; i < this.numHypotheses; i++) {
      var a = Math.random() * 4 - 2;
      var b = Math.random() * 4 - 2;
      var c = Math.random() * 4 - 2;

      currInfoGain = this.getInfoGain(examples, a, b, c);
      if (currInfoGain[0] > maxInfoGain[0]) {
        maxInfoGain = currInfoGain;
        besta = a;
        bestb = b;
        bestc = c;
      }
    }

    var examples_L = maxInfoGain[1];
    var examples_R = maxInfoGain[2];

    this.data[idx] = {
      isLeaf: false,
      threshold: [besta, bestb, bestc],
      label: 0
    };
    var leafNode = {
      isLeaf: true,
      threshold: [],
      label: this.getVote(examples)
    };

    if (examples_L.length == 0) {
      this.data[2 * (idx + 1) - 1] = leafNode;
    } else {
      this.train(2 * (idx + 1) - 1, examples_L);
    }
    if (examples_R.length == 0) {
      this.data[2 * (idx + 1)] = leafNode;
    } else {
      this.train(2 * (idx + 1), examples_R);
    }
  };

  this.isLeaf = function(idx) {
    var n_leaves = Math.pow(2, this.maxDepth);
    return this.data[idx].isLeaf || idx >= (this.size - n_leaves);
  }

  // get the majority vote of a set of examples
  this.getVote = function(examples) {
    var label = 0;
    var n_neg = 0;
    var n_pos = 0;
    for (var i = 0; i < examples.length; i++) {
      if (examples[i].label == 1) {
        n_pos++;
      } else {
        n_neg++;
      }
    }
    if (n_neg > n_pos) {
      label = -1;
    } else {
      label = 1;
    }
    return label;
  };

  // check if all examples in a set have the same label
  this.allSame = function(examples) {
    var same = true;
    var N = examples.length;

    if (N == 0) {
      return;
    }

    var label = examples[0].label;
    for (var i = 1; i < N; i++) {
      if (examples[i].label != label) {
        same = false;
        break;
      }
    }
    return same;
  }

  // print out the values in bfs order
  this.print = function() {
    for (var i = 0; i < this.size; i++) {
      console.log(this.data[i]);
    }
  };

  // predict the binary value for pixel x, y
  this.predict = function(x, y) {
    var idx = 0;
    var goLeft = true;
    while (!this.isLeaf(idx)) {
      var isLead = this.isLeaf(idx);
      var t = this.data[idx].threshold;

      var a = t[0];
      var b = t[1];
      var c = t[2];
      var val = a * x + b * y + c;
      if (val > 0) {
        goLeft = false;
      }
      if (goLeft) {
        idx = 2 * (idx + 1) - 1;
      } else {
        idx = 2 * (idx + 1);
      }
    }
    return this.data[idx].label;
  }

  // calculate the entropy of two sets of examples
  this.getEntropy = function(examples) {
    var entropy = 0;
    // partition into positive and negative examples
    var pos_examples = new Array();
    var neg_examples = new Array();
    var N = examples.length;
    for (var i = 0; i < N; i++) {
      if (examples[i].label == 1) {
        pos_examples.push(examples[i]);
      } else {
        neg_examples.push(examples[i]);
      }
    }
    var n_pos = pos_examples.length;
    var n_neg = neg_examples.length;
    var p_pos = n_pos / N;
    var p_neg = n_neg / N;
    if (p_pos == 1 || p_neg == 1) {
      entropy = 0;
    } else {
      entropy = -p_pos * Math.log(p_pos) - p_neg * Math.log(p_neg);
    }
    return entropy;
  }

  // calculate the information gain arising from partioning
  // examples in a certain way
  this.getInfoGain = function(examples, a, b, c) {
    var N = examples.length;
    var entropyBefore = this.getEntropy(examples);
    var lrExamples = this.partition(examples, a, b, c);
    var examples_left = lrExamples[0];
    var examples_right = lrExamples[1];

    var entropyLeft = this.getEntropy(examples_left);
    var entropyRight = this.getEntropy(examples_right);

    var n_left = examples_left.length;
    var n_right = examples_right.length;

    var entropyAfter = (n_left / N) * entropyLeft + (n_right / N) *
      entropyRight;
    var gain = entropyBefore - entropyAfter;

    return [gain, examples_left, examples_right];
  }

  // partition examples into left and right
  // by thresholding a particular variable
  this.partition = function(examples, a, b, c) {
    var left_examples = new Array();
    var right_examples = new Array();
    var N = examples.length;
    for (var i = 0; i < N; i++) {
      //var val = examples[i][Object.keys(examples[i])[1]];
      var x = examples[i].x;
      var y = examples[i].y;
      var val = a * x + b * y + c;
      if (val <= 0) {
        left_examples.push(examples[i]);
      } else {
        right_examples.push(examples[i]);
      }
    }
    return [left_examples, right_examples];
  }
}
// END OF DECISIONTREE
