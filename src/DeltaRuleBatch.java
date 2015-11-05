import java.util.List;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class DeltaRuleBatch extends DeltaRule {
    /* sum of delta weight */
    List<Double> sumOfDeltaWeight;

    @Override
    public double computeDeltaWeight() {
        return 0;
    }

    @Override
    public double computeEpochError() {
        return 0;
    }
}
