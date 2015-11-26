import weka.core.Instances;

import java.util.List;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class PerceptronTrainingRule extends DeltaRule{

    @Override
    public Instances readInput(String filename) {
        return null;
    }

    @Override
    public void loadInstancesIntoInputValue(Instances instances) {

    }

    @Override
    public void loadOrGenerateInputWeight(boolean isRandom) {

    }

    @Override
    public void loadTargetFromInstances(Instances instances) {

    }

    @Override
    public double computeEpochError(List<Double> finalErrorThisEpoch) {
        return 0;
    }

    @Override
    public double computeOutputInstance(Double[] inputValueThisInstance, Double[] inputWeightThisInstance) {
        return 0;
    }

    @Override
    public Double[] computeDeltaWeightInstance(Double[] inputValueThisInstance, double errorThisInstance, int indexData) {
        return new Double[0];
    }

    @Override
    public Double[] computeNewWeightInstance(Double[] inputWeightThisInstance, Double[] deltaWeightThisInstance) {
        return new Double[0];
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

    }
}
