import Util.Util;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        Util.loadARFF("data/simple.weather.arff");
        Util.buildModel("mlp");
    }
}
