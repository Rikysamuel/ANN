import java.util.List;

/**
 * Abstract class for Delta Rule Batch and Incremental
 */
public abstract class DeltaRule {
    /* the learning rate */
    Double learningRate;
    /* threshold */
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

    /* compute the delta weight of a weight */
    public abstract double computeDeltaWeight();

    /* compute the error of Epoch */
    public abstract double computeEpochError();

}
