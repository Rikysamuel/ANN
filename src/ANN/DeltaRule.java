package ANN;

import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract class for Delta Rule Batch and Incremental
 */
public abstract class DeltaRule extends Classifier {
    /* the learning rate */
    Double learningRate;
    /* maximum epoch */
    int maxEpoch;
    /* threshold error */
    Double threshold;
    /* momentum */
    Double momentum;
    /* vector of input value for neuron */
    List<Double[]> inputValue;
    /* vector of input weight for neuron */
    List<List<Double[]>> inputWeight;
    /* vector of target value */
    List<List<Double>> target;
    /* the result of net function */
    List<List<Double>> output;

    /* the difference between target and output (t-o) */
    List<Double> errorToTarget;
    /* the delta weight */
    List<List<Double[]>> deltaWeight;
    /* the new weight */
    List<List<Double[]>> newWeight;

    /* dataset input weka */
    Instances inputDataSet;
    /* number of data */
    int numData;
    /* number of attributes */
    int numAttributes;
    /* number of class label */
    int numClasses;

    /* check if iteration is convergent or not */
    boolean isConvergent;

    /* default constructor */
    public DeltaRule() {
        learningRate = 0.1;
        maxEpoch = 5;
        threshold = 0.0001;
        momentum = 0.2;
        isConvergent = false;
        inputValue = new ArrayList<>();
        inputWeight = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        errorToTarget = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
    }

    /* constructor */
    public DeltaRule(Double learningRate,int maxEpoch,Double threshold,Double momentum) {
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.threshold = threshold;
        this.momentum = momentum;
        isConvergent = false;
        inputValue = new ArrayList<>();
        inputWeight = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        errorToTarget = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        for (List<Double[]> listDeltaWeight : deltaWeight) {
            listDeltaWeight = new ArrayList<>();
        }
        newWeight = new ArrayList<>();
        for (List<Double[]> listNewWeight : newWeight) {
            listNewWeight = new ArrayList<>();
        }
    }

    /* Initialize final delta weight resulted per epoch */
    public abstract void initializeFinalDeltaWeight();

    /* Initialize final new weight resulted per epoch */
    public abstract void initializeFinalNewWeight();

    /* Convert instances data into input value */
    public abstract void loadInstancesIntoInputValue(Instances instances);

    /* Randomize or generalize weight each input */
    public abstract void loadOrGenerateInputWeight(boolean isRandom);

    /* Load target value from arff file */
    public abstract void loadTargetFromInstances(Instances instances);

    /* compute the error of Epoch */
    public abstract double computeEpochError(List<Double> finalErrorThisEpoch);

    /* compute output of one instance using sigmoid activation function */
    public abstract Double computeOutputInstance(Double[] inputValueThisInstance, Double[] inputWeightThisInstance);

    /* compute delta weight of one instance */
    public abstract Double[] computeDeltaWeightInstance(Double[] inputValueThisInstance, Double errorThisInstance, int indexData, int neuronOutputIndex);

    /* compute new weight yields for one instance */
    public abstract Double[] computeNewWeightInstance(Double[] inputWeightThisInstance, Double[] deltaWeightThisInstance);

    /* compute error (target - output) each output neuron */
    public abstract Double computeErrorThisInstance(Double targetOutputPerNeuron, Double outputPerNeuron);
}
