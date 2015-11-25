import weka.core.Instances;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class DeltaRuleIncremental extends DeltaRule {

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
    public double computeDeltaWeight() {
        return 0;
    }

    @Override
    public double computeEpochError() {
        return 0;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

    }
}
