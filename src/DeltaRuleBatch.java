import weka.core.Instances;

import java.util.List;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class DeltaRuleBatch extends DeltaRule {
    /* dataset input weka */
    Instances inputDataSet;

    /* Konstruktor */
    public DeltaRuleBatch(Double learningRate,int maxEpoch,Double threshold) {
        super(learningRate,maxEpoch,threshold);
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
