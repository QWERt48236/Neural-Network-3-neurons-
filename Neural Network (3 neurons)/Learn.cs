using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network__3_neurons_
{
    public class NeuralNetwork
    {
        double w000 = 0.2;
        double w010 = -0.8;
        double w001 = 0.4;
        double w011 =- 0.6;
        double w10 = 0.3;
        double w11 = -0.7;


        double NeuronActivation(double x1, double x2, double w1, double w2) 
        {
            return 1 / (1 + Math.Exp(-(x1 * w1 + x2 * w2)));
        }

        double ExitNeuronSum(double x1, double x2)
        {
            double neuron1 = 1 / (1 + Math.Exp(-(x1 * w000 + x2 * w010)));
            double neuron2 = 1 / (1 + Math.Exp(-(x1 * w001 + x2 * w011)));

            return (neuron1 * w10 + neuron2 * w11);
        }

        double ForwardPropagation(double x1, double x2)
        {
            double neuron1 = 1 / (1 + Math.Exp(-(x1 * w000 + x2 * w010)));
            double neuron2 = 1 / (1 + Math.Exp(-(x1 * w001 + x2 * w011)));

            double y = 1 / (1 + Math.Exp(-(neuron1 * w10 + neuron2 * w11)));

            return y;
        }

        void UpdateWeights(double Yexpected, double x1, double x2, double learningRate)
        {
            w000 = w000 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * ForwardPropagation(x1, x2) * NeuronActivation(x1, x2, w000, w010) * Math.Pow(ForwardPropagation(x1, x2), 2) * (-w10) * x1);
            w010 = w010 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * ForwardPropagation(x1, x2) * NeuronActivation(x1, x2, w000, w010) * Math.Pow(ForwardPropagation(x1, x2), 2) * (-w10) * x2);
            w001 = w001 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * ForwardPropagation(x1, x2) * NeuronActivation(x1, x2, w001, w011) * Math.Pow(ForwardPropagation(x1, x2), 2) * (-w11) * x1);
            w011 = w011 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * ForwardPropagation(x1, x2) * NeuronActivation(x1, x2, w001, w011) * Math.Pow(ForwardPropagation(x1, x2), 2) * (-w11) * x2);
            w10 = w10 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * ForwardPropagation(x1, x2) * NeuronActivation(x1, x2, w000, w010) * Math.Pow(ForwardPropagation(x1, x2), 2));
            w11 = w11 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * ForwardPropagation(x1, x2) * NeuronActivation(x1, x2, w001, w011) * Math.Pow(ForwardPropagation(x1, x2), 2));
        }

        double NetworkErrorFunction(double[] x1, double[] x2, double[] Yexpected)
        {
            double sum = 0;

            for (int i = 0; i < x1.Length; i++)
            {
                sum += ForwardPropagation(x1[i], x2[i]) - Yexpected[i];
                Console.WriteLine(ForwardPropagation(x1[i], x2[i]) - Yexpected[i]);
            }

            return sum/ x1.Length;
        }

        public void Learn(double[] x1, double[] x2, double[] Yexpected, double learningRate, int epochs)
        {
            Console.WriteLine($"Похибка мережі до навчання: {NetworkErrorFunction(x1, x2, Yexpected)}");

            for (int i = 0; i < epochs; i++)
            {
                for(int j = 0; j < 1; j++)
                {
                    UpdateWeights(Yexpected[j], x1[j], x2[j], learningRate);

                }
            }

            Console.WriteLine($"Похибка мережі після навчання: {NetworkErrorFunction(x1, x2, Yexpected)}");
        }
    }
}