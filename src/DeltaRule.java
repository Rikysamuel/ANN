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
    /* vector of input value for neuron */
    List<Double[]> inputValue;
    /* vector of input weight for neuron */
    List<Double[]> inputWeight;
    /* vector of target value */
    List<Double> target;
    /* the result of net function */
    List<Double> output;

    /* the difference between target and output (t-o) */
    List<Double> errorToTarget;
    /* the delta weight */
    List<Double[]> deltaWeight;
    /* the new weight */
    List<Double[]> newWeight;

    /* default constructor */
    public DeltaRule() {
        learningRate = 0.1;
        maxEpoch = 5;
        threshold = 0.0001;
        inputValue = new ArrayList<>();
        inputWeight = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        errorToTarget = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
    }

    /* constructor */
    public DeltaRule(Double learningRate,int maxEpoch,Double threshold) {
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.threshold = threshold;
        inputValue = new ArrayList<>();
        inputWeight = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        errorToTarget = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
    }

    /* Load instances from arff file */
    public abstract Instances readInput(String filename);

    /* Convert instances data into input value */
    public abstract void loadInstancesIntoInputValue(Instances instances);

    /* Randomize or generalize weight each input */
    public abstract void loadOrGenerateInputWeight(boolean isRandom, Double[] weight);

    /* compute the delta weight of a weight */
    public abstract double computeDeltaWeight();

    /* compute the error of Epoch */
    public abstract double computeEpochError();
}
