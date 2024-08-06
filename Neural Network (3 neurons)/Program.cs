

using Neural_Network__3_neurons_;

double[] x1 = { 0.5, 0.4, 0.5, 0.9, 0.5 };
double[] x2 = { 0.5, 0.5, 0.3, 0.5, 0.1 };

double[] Yexpected = { 1, 0, 1, 1, 1 };

double learningRate = 0.1;

NeuralNetwork neural = new NeuralNetwork();

neural.Learn(x1, x2, Yexpected, learningRate, 10);


