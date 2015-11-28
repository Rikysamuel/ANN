package Util;

import java.io.Serializable;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Neuron class helper
 * Created by rikysamuel on 11/20/2015.
 */
public class Neuron implements Serializable{
    private static final long serialVersionUID = -5990607817048210779L;
    public double outValue;
    public SortedMap<Integer, Double> input;
    public double error;

    /* if output neuron */
    public SortedMap<Integer, Double> targetValue;

    public Neuron() {
    }

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
