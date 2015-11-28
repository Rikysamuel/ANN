import Util.Util;
import weka.core.Instances;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        Util.loadARFF("D:\\weka-3-6\\data\\iris.arff");
//        Util.loadARFF("data/train.arff");
        Util.buildModel("incremental");

//        Util.loadARFF("data/test.weather.numeric.arff");
//        Instances test = Util.setNominalToBinary(Util.getData());
        Util.FullSchema(Util.getData());
    }
}
