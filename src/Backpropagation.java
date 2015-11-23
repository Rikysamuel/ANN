import Util.ActivationClass;
import Util.Neuron;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class BackPropagation extends Classifier {
    public Instances data;
    public double learningRate;
    public double momentum;
    public double biasWeight;
    public Map<String, List<Integer>> neuronType;
    public int numOfHiddenNeuron;
    public int numOfOutputNeuron;
    public Map<Integer, Neuron> neurons;
    public Map<Integer, Map<Integer, Double[]>> weights;
    public Map<String, Double> classValue;

    public BackPropagation(){
        learningRate = 0.1;
        momentum = 0.1;
        numOfHiddenNeuron = 2;

        neurons = new HashMap<>();
        weights = new HashMap<>();
        classValue = new HashMap<>();
        neuronType = new HashMap<>();
    }

    public void setBiasValue(double biasValue) {
        SortedMap<Integer, Double> temp = new TreeMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            temp.put(i, biasValue);
        }
        neurons.put(0, new Neuron(temp, 0.0));
    }

    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    public void setNumOfInputNeuron() {
        Map<Integer, SortedMap<Integer, Double>> attribData = parseData();
        int numOfInput = data.numAttributes();
        int counter = 1;

        while(counter <= numOfInput) {
            neurons.put(counter-1, new Neuron(attribData.get(counter-1), 0));
            counter++;
        }
    }

    public void setNumNeuron(int numNeuron, boolean isHidden) {
        String type;
        int last = neurons.size() - 1;
        Map<Integer, Double[]> temp;
        Double[] d = new Double[3];
        List<Integer> vals = new ArrayList<>();
        d[0] = biasWeight;
        d[1] = 0.0; // ignore this
        d[2] = 0.0; // ignore this

        if (isHidden) {
            this.numOfHiddenNeuron = numNeuron;
            type = "hidden";
        } else {
            this.numOfOutputNeuron = numNeuron;
            type = "output";
        }

        int counter = last + 1;
        while(counter <= last + numNeuron) {
            if (weights.containsKey(counter)) {
                temp = weights.get(counter);
            } else {
                temp = new HashMap<>();
            }

            temp.put(0, d);
            neurons.put(counter, new Neuron(0));
            weights.put(counter, temp);
            vals.add(counter);
            counter++;
        }
        neuronType.put(type, vals);
    }

    public Map<Integer, SortedMap<Integer, Double>> parseData() {
        Map<Integer, SortedMap<Integer, Double>> parsedData = new HashMap<>();
        SortedMap<Integer, Double> attributeData;
        for (int i = 0; i < data.numAttributes(); i++) {
            attributeData = new TreeMap<>();
            for (int j = 0; j < data.numInstances(); j++) {
                attributeData.put(j, data.instance(j).value(i));
            }
            parsedData.put(i + 1, attributeData);
        }

        return parsedData;
    }

    public void setNeuronConnectivity() {
        Map<Integer, Double[]> temp;
        Double[] d = new Double[3];
        d[0] = 0.1; // new weight
        d[1] = 0.0; // old weight
        d[2] = 0.0; // delta weight

        // input to hidden
        for (int i = 1; i < data.numAttributes(); i++) {
            for (int j = data.numAttributes(); j < data.numAttributes() + numOfHiddenNeuron; j++) {
                if (weights.containsKey(j)) {
                    temp = weights.get(j);
                } else {
                    temp = new HashMap<>();
                }
                temp.put(i,d);
                weights.put(j, temp);
            }
        }

        // hidden to output
        for (int i = data.numAttributes(); i < data.numAttributes() + numOfHiddenNeuron; i++) {
            for (int j = data.numAttributes() + numOfHiddenNeuron; j < data.numAttributes() + numOfHiddenNeuron + numOfOutputNeuron; j++) {
                if (weights.containsKey(j)) {
                    temp = weights.get(j);
                } else {
                    temp = new HashMap<>();
                }
                temp.put(i, d);
                weights.put(j, temp);
            }
        }
    }

    public void readInput(String filename) {
        FileReader file = null;
        try {
            file = new FileReader(filename);
            try (BufferedReader reader = new BufferedReader(file)) {
                data = new Instances(reader);
            }
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
            numOfOutputNeuron = data.numClasses();
        } catch (IOException ex) {
            Logger.getLogger(BackPropagation.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (file!=null) {
                    file.close();
                }
                setNumOfInputNeuron();
            } catch (IOException ex) {
                Logger.getLogger(BackPropagation.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    public Map<Integer, Double[]> findNodeBefore(int node) {
        return weights.get(node);
    }

    public Map<Integer, Double[]> findNodeAfter(int node) {
        Map<Integer, Double[]> temp = new HashMap<>();

        for (Map.Entry<Integer, Map<Integer, Double[]>> weight : weights.entrySet()) {
            weight.getValue().entrySet().stream().filter(refs -> refs.getKey() == node).forEach(refs ->
                    temp.put(weight.getKey(), refs.getValue()));
        }

        return temp;
    }

    public double findWeight(int node1, int node2) {
        return weights.get(node2).get(node1)[0];
    }

    public double computeOutputValue(int node, int instanceNo) {
        double temp = 0;
        Map<Integer, Double[]> refs = findNodeBefore(node);
        Map<Integer, Double[]> refsAfter = findNodeAfter(node);

        for (Map.Entry<Integer, Double[]> ref : refs.entrySet()) {
            temp += findWeight(ref.getKey(), node) * neurons.get(ref.getKey()).input.get(instanceNo);
        }

        Neuron n = new Neuron(neurons.get(node));
        n.outValue = temp;

        for (int i = 0; i < data.numInstances(); i++) {
            n.input.put(i, ActivationClass.sigmoid(temp));
        }
        neurons.put(node, n);

        for (Map.Entry<Integer, Double[]> ref : refsAfter.entrySet()) {
            n = new Neuron(neurons.get(ref.getKey()));
            n.input.put(0, ActivationClass.sigmoid(temp));
            neurons.put(ref.getKey(), n);
        }

        return temp;
    }

    public void setNominalToBinary() {
        NominalToBinary ntb = new NominalToBinary();
        try {
            ntb.setInputFormat(data);
            data = new Instances(Filter.useFilter(data, ntb));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * assign instance target value to neuron
     * @param instanceNo instance number
     */
    public void setInstanceTarget(int instanceNo) {
        List<Integer> outNeuron = neuronType.get("output");
        Neuron n;
        for (int i = 0; i < outNeuron.size(); i++) {
            n = neurons.get(outNeuron.get(i));
            if (Double.compare((double) i, data.instance(instanceNo).classValue()) == 0) {
                n.targetValue.put(instanceNo, 1.0);
            } else {
                n.targetValue.put(instanceNo, 0.0);
            }
            neurons.put(outNeuron.get(i), n);
        }

    }

    public void computeOutputNeuronError(int instanceNo) {
        List<Integer> outNeuron = neuronType.get("output");
        Neuron n;
        for (int i = 0; i < outNeuron.size(); i++) {
            n = neurons.get(outNeuron.get(i));
            n.error = n.input.get(0) * (1 - n.input.get(0)) * (n.targetValue.get(instanceNo) - n.input.get(0));
            neurons.put(outNeuron.get(i), n);
        }
    }

    public void computeHiddenNeuronError(int instanceNo) {
        List<Integer> hiddenNeuron = neuronType.get("hidden");
        Map<Integer, Double[]> nodeAfter;
        double temp;

        Neuron n;
        for (int i = 0; i < hiddenNeuron.size(); i++) {
            n = neurons.get(hiddenNeuron.get(i));
            n.error = n.input.get(instanceNo) * (1 - n.input.get(instanceNo));

            nodeAfter = findNodeAfter(hiddenNeuron.get(i));
            temp = 0;
            for (Map.Entry<Integer, Double[]> outNode : nodeAfter.entrySet()) {
                temp += (neurons.get(outNode.getKey()).error * findWeight(hiddenNeuron.get(i), outNode.getKey()));
            }
            n.error *= temp;

            neurons.put(hiddenNeuron.get(i), n);
        }
    }

    public void updateWeights(int instanceNo) {
        Double[] tempDouble;
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                tempDouble = new Double[3];
                System.out.println(weight.getKey() + " " + realWeight.getKey());
                System.out.println(neurons.get(realWeight.getKey()).input.get(instanceNo));
                System.out.println(neurons.get(weight.getKey()).error);
                double d = realWeight.getValue()[0];
                System.out.println("old: " + d);
                tempDouble[2] = learningRate *
                        neurons.get(realWeight.getKey()).input.get(instanceNo) *
                        neurons.get(weight.getKey()).error + (momentum * realWeight.getValue()[2]); // compute delta weight
                System.out.println("delta: " + tempDouble[2]);
                tempDouble[1] = realWeight.getValue()[2]; // hold the old delta weight
                tempDouble[0] = d + tempDouble[2];
                realWeight.setValue(tempDouble);
            }
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

    }
}
