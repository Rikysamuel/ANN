package ANN;

import Util.ActivationClass;
import Util.Neuron;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
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
 * Backpropagation Algorithm for ANN classifier
 * Created by rikysamuel on 11/4/2015.
 */
public class Backpropagation extends Classifier {
    private int numEpoch;
    private int numOfHiddenNeuron;
    private int numOfOutputNeuron;
    private double learningRate;
    private double momentum;
    private double biasWeight;
    private double MSEThreshold;

    public Instances data;
    private Map<String, List<Integer>> neuronType;
    private Map<Integer, Neuron> neurons;
    private Map<Integer, Map<Integer, Double[]>> weights;

    public Backpropagation(){
        learningRate = 0.1;
        momentum = 0.1;
        numOfHiddenNeuron = 2;

        neurons = new HashMap<>();
        weights = new HashMap<>();
        neuronType = new HashMap<>();
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getNumEpoch() {
        return numEpoch;
    }

    public void setNumEpoch(int numEpoch) {
        this.numEpoch = numEpoch;
    }

    public void setBiasValue(double biasValue) {
        System.out.println(data);
        SortedMap<Integer, Double> temp = new TreeMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            temp.put(i, biasValue);
        }

        System.out.println("asd: " + temp);
        neurons.put(0, new Neuron(temp, 0.0));
    }

    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    public void setNumOfInputNeuron() {
        numOfOutputNeuron = data.numClasses();
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

    public Instances readInput(String filename) {
        FileReader file = null;
        try {
            file = new FileReader(filename);
            try (BufferedReader reader = new BufferedReader(file)) {
                data = new Instances(reader);
            }
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
        } catch (IOException ex) {
            Logger.getLogger(Backpropagation.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (file!=null) {
                    file.close();
                }
                setNumOfInputNeuron();
            } catch (IOException ex) {
                Logger.getLogger(Backpropagation.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        return data;
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
        for (Integer anOutNeuron : outNeuron) {
            n = neurons.get(anOutNeuron);
            n.error = n.input.get(0) * (1 - n.input.get(0)) * (n.targetValue.get(instanceNo) - n.input.get(0));
            neurons.put(anOutNeuron, n);
        }
    }

    public void computeHiddenNeuronError(int instanceNo) {
        List<Integer> hiddenNeuron = neuronType.get("hidden");
        Map<Integer, Double[]> nodeAfter;
        double temp;

        Neuron n;
        for (Integer aHiddenNeuron : hiddenNeuron) {
            n = neurons.get(aHiddenNeuron);
            n.error = n.input.get(instanceNo) * (1 - n.input.get(instanceNo));

            nodeAfter = findNodeAfter(aHiddenNeuron);
            temp = 0;
            for (Map.Entry<Integer, Double[]> outNode : nodeAfter.entrySet()) {
                temp += (neurons.get(outNode.getKey()).error * findWeight(aHiddenNeuron, outNode.getKey()));
            }
            n.error *= temp;

            neurons.put(aHiddenNeuron, n);
        }
    }

    public void updateWeights(int instanceNo) {
        Double[] tempDouble;
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                tempDouble = new Double[3];
                double d = realWeight.getValue()[0];
                tempDouble[2] = learningRate *
                        neurons.get(realWeight.getKey()).input.get(instanceNo) *
                        neurons.get(weight.getKey()).error + (momentum * realWeight.getValue()[2]); // compute delta weight
                tempDouble[1] = realWeight.getValue()[2]; // hold the old delta weight
                tempDouble[0] = d + tempDouble[2];
                realWeight.setValue(tempDouble);
            }
        }
    }

    public void printNeuron() {
        System.out.println("Neuron - Input Value (Activation) - Target - Net Value - Error");
        for (Map.Entry<Integer, Neuron> neuron : neurons.entrySet()) {
            System.out.print(neuron.getKey() + " ");
            System.out.print(neuron.getValue().input);
            System.out.print(" "+ neuron.getValue().targetValue + " ");
            System.out.print(" [" + neuron.getValue().outValue + "] ");
            System.out.println(" [" + neuron.getValue().error + "] ");
        }
    }

    public void printWeight() {
        System.out.println();
        System.out.println("Connection - New Weight - Old Weight - Delta Weight");
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                System.out.print(realWeight.getKey() + "-" + weight.getKey() + " ");
                System.out.print(realWeight.getValue()[0] + " ");
                System.out.print(realWeight.getValue()[1] + " ");
                System.out.println(realWeight.getValue()[2] + " ");
            }
        }
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        instances = new Instances(instances);
        setNumNeuron(instances.numClasses(), false); //output
        setNeuronConnectivity();

        List<Integer> hidden = neuronType.get("hidden");
        System.out.println(hidden);
        List<Integer> output = neuronType.get("output");
        System.out.println(output);

        for (int epoch = 0; epoch < numEpoch; epoch++) {
            for (int i = 0; i < instances.numInstances(); i++) {
                setInstanceTarget(i);

                for (Integer aHidden : hidden) {
                    computeOutputValue(aHidden, i);
                }
                for (Integer anOutput : output) {
                    computeOutputValue(anOutput, i);
                }

                computeOutputNeuronError(i);
                computeHiddenNeuronError(i);

                updateWeights(i);

                for (Integer aHidden : hidden) {
                    computeOutputValue(aHidden, i);
                }
                for (Integer anOutput : output) {
                    computeOutputValue(anOutput, i);
                }

                printNeuron();
                printWeight();
            }
        }
    }

    public double classifyInstance(Instance instance) {
        List<Integer> hiddNeuron = neuronType.get("hidden");
        List<Integer> outNeuron = neuronType.get("output");
        Map<Integer, Double[]> neuronBefore;
        double weight;



        /*for (int i = 0; i < hiddNeuron.size(); i++) {

            weight = 0;
            neuronBefore = findNodeBefore(hiddNeuron.get(i));

            for (int j = 0; j <= neuronBefore.size(); j++) {
                weight += weights.get(hiddNeuron.get(i)).get(j)[0] * instance.value(j - 1);
            }
            neurons.get(hiddNeuron.get(i)).input = new TreeMap<>();
            neurons.get(hiddNeuron.get(i)).input.put(hiddNeuron.get(i), weight);
        }

        for (int i = 0; i < outNeuron.size(); i++) {

        }*/

        return 0;
    }
}
