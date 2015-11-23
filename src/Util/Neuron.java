package Util;

import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Neuron {
    public double outValue;
    public SortedMap<Integer, Double> input;
    public double error;

    /* if output neuron */
    public SortedMap<Integer, Double> targetValue;

    public Neuron(Neuron neuron) {
        this.outValue = neuron.outValue;
        this.input = neuron.input;
        this.error = neuron.error;
        this.targetValue = neuron.targetValue;
    }

    public Neuron(double outValue) {
        this.outValue = outValue;
        input = new TreeMap<>();
        targetValue = new TreeMap<>();
    }

    public Neuron(SortedMap<Integer, Double> input, double outValue) {
        this.input = input;
        this.outValue = outValue;
    }
}
