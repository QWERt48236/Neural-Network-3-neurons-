using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network__3_neurons_
{
    public class NeuralNetwork
    {
        double w000 = 0.7;
        double w010 = 0.3;
        double w001 = 0.6;
        double w011 = 0.4;
        double w10 = 0.8;
        double w11 = 0.2;

        double ExitNeuronSum(double x1, double x2)
        {
            return ((x1 * w000 + x2 * w010) * w10 + (x1 * w001 + x2 * w011) * w11);
        }

        double ForwardPropagation(double x1, double x2)
        {
            double y = 1 / (1 + Math.Exp(-ExitNeuronSum(x1, x2)));

            return y;
        }

        void UpdateWeights(double Yexpected, double x1, double x2, double learningRate)
        {
            w000 = w000 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * Math.Pow(ForwardPropagation(x1, x2), 2) * Math.Exp(ExitNeuronSum(x1, x2)) * w10 * x1);
            w010 = w010 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * Math.Pow(ForwardPropagation(x1, x2), 2) * Math.Exp(ExitNeuronSum(x1, x2)) * w10 * x2);
            w001 = w001 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * Math.Pow(ForwardPropagation(x1, x2), 2) * Math.Exp(ExitNeuronSum(x1, x2)) * w11 * x1);
            w011 = w011 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * Math.Pow(ForwardPropagation(x1, x2), 2) * Math.Exp(ExitNeuronSum(x1, x2)) * w11 * x2);
            w10 = w10 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * Math.Pow(ForwardPropagation(x1, x2), 2) * Math.Exp(ExitNeuronSum(x1, x2)) * (-(x1 * w000 + x2 * w010)));
            w11 = w11 - learningRate * (2 * (ForwardPropagation(x1, x2) - Yexpected) * Math.Pow(ForwardPropagation(x1, x2), 2) * Math.Exp(ExitNeuronSum(x1, x2)) * (-(x1 * w001 + x2 * w011)));
        }

        double NetworkErrorFunction(double[] x1, double[] x2, double[] Yexpected)
        {
            double sum = 0;

            for (int i = 0; i < 1; i++)
            {
                sum += ForwardPropagation(x1[i], x2[i]) - Yexpected[i];
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