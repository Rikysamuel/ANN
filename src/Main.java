import Util.Neuron;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        Backpropagation bp = new Backpropagation();
        Instances data = bp.readInput("data/simple.weather.arff");

        /* config */
        bp.setNominalToBinary();
        bp.setBiasValue(1);
        bp.setNumNeuron(2, true); //hidden
        bp.setMomentum(0.1);
        bp.setLearningRate(0.1);
        bp.setNumEpoch(4);

        bp.buildClassifier(data);
    }
}
