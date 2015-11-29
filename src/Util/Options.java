package Util;

/**
 * Helper class to define the topology of the ANN
 * Created by rikysamuel on 11/27/2015.
 */
public class Options {
    /* the bias value */
    public static double biasValue;
    /* the bias weight */
    public static double biasWeight;
    /* the weight initial value, -1 means random to MLP */
    public static double initWeight = -1;
    /* num of hidden neuron, if algorithm used is multi layer perceptron*/
    public static int numOfHiddenNeuron;
    /* momentum value */
    public static double momentum;
    /* learning rate value */
    public static double learningRate;
    /* num of max epoch */
    public static int maxEpoch;
    /* the MSE threshold */
    public static double MSEthreshold;
    /*Activation function*/
    public static String activationFunction;
}
