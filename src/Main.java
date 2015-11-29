import Util.Util;
import Util.Options;
import weka.core.Instances;

/**
 * Main class 
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
        Options.MSEthreshold = 0.3387570349024238;
        Options.function = "sigmoid";

//        Util.loadARFF("data/simple.weather.arff");
 //       Util.loadARFF("data/weather.nominal.arff");
//        Util.loadARFF("data/weather.numeric.arff");
//        Util.loadARFF("data/iris.arff");
//        Util.loadARFF("data/iris.2D.arff");

     //   Util.loadARFF("D:\\weka-3-6\\data\\weather.nominal.arff");
     //   Util.loadARFF("D:\\weka-3-6\\data\\weather.numeric.arff");
     //   Util.loadARFF("D:\\weka-3-6\\data\\iris.arff");
        Util.loadARFF("D:\\weka-3-6\\data\\iris.2D.arff");

        Instances data = Util.setNominalToBinary(Util.getData());
        data = Util.useNormalization(data);
        Util.setData(data);

        Util.buildModel("batch");

      //  Util.FullSchema(data);
        Util.FoldSchema(data, 10);
    //    Util.PercentageSplit(data, 66.67, "batch");
    }
}
