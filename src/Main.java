import Util.Util;
import Util.Options;
import weka.core.Instances;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        Options.biasValue = 1;
        Options.biasWeight = 0.1;
        Options.initWeight = 0.1;
        Options.numOfHiddenNeuron = 2;
        Options.momentum = 0.1;
        Options.learningRate = 0.1;
        Options.maxEpoch = 1000;
//        Options.MSEthreshold = 0.3387570349024238;

//        Util.loadARFF("data/simple.weather.arff");
//        Util.loadARFF("data/weather.nominal.arff");
        Util.loadARFF("data/iris.arff");
//        Util.loadARFF("data/weather.numeric.arff");

        Instances data = Util.setNominalToBinary(Util.getData());
        data = Util.useNormalization(data);
        Util.setData(data);

        Util.buildModel("mlp");

        Util.FullSchema(data);
    }
}
