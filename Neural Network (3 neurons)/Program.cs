

using Neural_Network__3_neurons_;

double[] x1 = { 0.9, 0.2, 0.7, 0.9, 0.8, 0.2, 0.7 };
double[] x2 = { 0.1, 0.8, 0.3, 0.1, 0.2, 0.8, 0.3 };

double[] Yexpected = { 1, 0, 1, 1, 1, 0, 0, 0};

double learningRate = 0.1;

NeuralNetwork neural = new NeuralNetwork();

neural.Learn(x1, x2, Yexpected, learningRate, 1000);


