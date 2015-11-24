import weka.classifiers.Classifier;

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
    }

    /* constructor */
    public DeltaRule(Double learningRate,int maxEpoch,Double threshold) {
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.threshold = threshold;
    }

    /* compute the delta weight of a weight */
    public abstract double computeDeltaWeight();

    /* compute the error of Epoch */
    public abstract double computeEpochError();
}
