package Util;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class ActivationClass {

    public static double sigmoid(double value) {
        double exp = 1 + Math.exp(-value);
        return (exp > 0) ? (1 / exp) : 0;
    }

    public static double sign() {
        return -1;
    }
}
