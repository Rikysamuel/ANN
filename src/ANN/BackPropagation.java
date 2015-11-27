package ANN;

import Util.ActivationClass;
import Util.Neuron;
import Util.Options;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Multi Layer Perceptron Algorithm for ANN classifier
 * Created by rikysamuel on 11/4/2015.
 */
public class BackPropagation extends Classifier {
    private int numEpoch;
    private int numOfHiddenNeuron;
    private int numOfOutputNeuron;
    private double learningRate;
    private double momentum;
    private double biasWeight;
    private double initWeight;
    private double MSEThreshold;

    private Instances data;
    private List<Double> error;
    private Map<Integer, Double> classMap;
    private Map<String, List<Integer>> neuronType;
    private Map<Integer, Neuron> neurons;
    private Map<Integer, Map<Integer, Double[]>> weights;

    public BackPropagation(Instances instances){
        setData(instances);

        initWeight = Options.initWeight;
        numOfHiddenNeuron = Options.numOfHiddenNeuron;
        momentum = Options.momentum;
        learningRate = Options.learningRate;
        numEpoch = Options.maxEpoch;
        MSEThreshold = Options.MSEthreshold;

        neurons = new HashMap<>();
        weights = new HashMap<>();
        neuronType = new HashMap<>();

        setNumOfInputNeuron();
        setBiasValue(Options.biasValue);
        setBiasWeight(Options.biasWeight);
        setHiddenNeuron(numOfHiddenNeuron);
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        this.data = data;
    }

    /**
     * set the bias value, add to neurons container
     * @param biasValue
     */
    public void setBiasValue(double biasValue) {
        SortedMap<Integer, Double> temp = new TreeMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            temp.put(i, biasValue);
        }

        neurons.put(0, new Neuron(temp, 0.0));
    }

    public double getBiasWeight() {
        return biasWeight;
    }

    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    /**
     * configure the input neurons
     */
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

    /**
     * invoke setNumNeuron method for method call simplicity
     * @param numNeuron
     */
    public void setHiddenNeuron(int numNeuron) {
        setNumNeuron(numNeuron, true);
    }

    /**
     * configure the hidden or output neuron
     * @param numNeuron the number of neurons to be configured
     * @param isHidden true if layer is hidden, false if layer is output
     */
    private void setNumNeuron(int numNeuron, boolean isHidden) {
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

    /**
     * parse the data based on the attribute
     * @return parsed data based on the attribute
     */
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

    /**
     * map the output neuron to the class index
     */
    public void setClassMap() {
        classMap = new HashMap<>();
        List<Integer> out = neuronType.get("output");

        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            classMap.put(out.get(i), (double) i);
        }
    }

    /**
     * configure connection between neurons
     */
    public void setNeuronConnectivity() {
        Map<Integer, Double[]> temp;
        Double[] d = new Double[3];
        d[0] = initWeight; // new weight
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
                temp.put(i, d);
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

        setClassMap();
    }

    /**
     * find nodes that references to a certain node
     * @param node the node
     * @return list of nodes that are references to a node
     */
    public List<Integer> findNodeBefore(int node) {
        return new ArrayList<>(weights.get(node).keySet());
    }

    /**
     * find the next neuron
     * @param node the node
     * @return the referenced node
     */
    public List<Integer> findNodeAfter(int node) {
        List<Integer> temp = new ArrayList<>();

        for (Map.Entry<Integer, Map<Integer, Double[]>> weight : weights.entrySet()) {
            weight.getValue().entrySet().stream().filter(refs -> refs.getKey() == node).forEach(refs ->
                    temp.add(weight.getKey()));
        }

        return temp;
    }

    /**
     * find the weight between 2 nodes
     * @param node1 first node
     * @param node2 second node
     * @return the weight between nodes
     */
    public double findWeight(int node1, int node2) {
        return weights.get(node2).get(node1)[0];
    }

    /**
     * compute the output value of a neuron on the certain instance
     * @param node the neuron's id
     * @param instanceNo the instance
     * @return the net value of a neuron
     */
    public double computeOutputValue(int node, int instanceNo) {
        double temp = 0;
        List<Integer> refs = findNodeBefore(node);
        List<Integer> refsAfter = findNodeAfter(node);

        for (Integer ref : refs) {
            temp += findWeight(ref, node) * neurons.get(ref).input.get(instanceNo);
        }

        Neuron n = new Neuron(neurons.get(node));
        n.outValue = temp;

        for (int i = 0; i < data.numInstances(); i++) {
            n.input.put(i, ActivationClass.sigmoid(temp));
        }
        neurons.put(node, n);

        for (Integer ref : refsAfter) {
            n = new Neuron(neurons.get(ref));
            n.input.put(0, ActivationClass.sigmoid(temp));
            neurons.put(ref, n);
        }

        return temp;
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

    /**
     * compute the error of the output layer
     * @param instanceNo the instance related
     */
    public void computeOutputNeuronError(int instanceNo) {
        List<Integer> outNeuron = neuronType.get("output");
        Neuron n;
        for (Integer anOutNeuron : outNeuron) {
            n = neurons.get(anOutNeuron);
            n.error = n.input.get(0) * (1 - n.input.get(0)) * (n.targetValue.get(instanceNo) - n.input.get(0));
            neurons.put(anOutNeuron, n);
        }
    }

    /**
     * compute the error of the hidden layer
     * @param instanceNo the instance related
     */
    public void computeHiddenNeuronError(int instanceNo) {
        List<Integer> hiddenNeuron = neuronType.get("hidden");
        List<Integer> nodeAfter;
        double temp;

        Neuron n;
        for (Integer aHiddenNeuron : hiddenNeuron) {
            n = neurons.get(aHiddenNeuron);
            n.error = n.input.get(instanceNo) * (1 - n.input.get(instanceNo));

            nodeAfter = findNodeAfter(aHiddenNeuron);
            temp = 0;
            for (Integer outNode : nodeAfter) {
                temp += (neurons.get(outNode).error * findWeight(aHiddenNeuron, outNode));
            }
            n.error *= temp;

            neurons.put(aHiddenNeuron, n);
        }
    }

    /**
     * add error (t-o) to the error container
     * @param instanceNo instance related
     * @param diffTargetOutput target minus output
     */
    public void addError(int instanceNo, double diffTargetOutput) {
        error.add(instanceNo, diffTargetOutput);
    }

    /**
     * compute the Mean SquareRoot Error based on the equation: 0.5 * (sum (t-o)^2)
     * @return the Mean SquareRoot Error
     */
    public double computeMSE() {
        double temp = 0;
        for (Double anError : error) {
            temp += Math.pow(anError, 2);
        }

        return temp/2;
    }

    /**
     * update the weight of each neuron to another neurons
     * @param instanceNo the related instance
     */
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

    /**
     * print the neurons data
     */
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

    /**
     * print the neuron's weights data
     */
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
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        return result;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        setNumNeuron(instances.numClasses(), false); //output
        setNeuronConnectivity();

        double temp, max = 0; int biggestOutNeuron = -1;
        List<Integer> hidden = neuronType.get("hidden");
        List<Integer> output = neuronType.get("output");

        for (int epoch = 0; epoch < numEpoch; epoch++) {
            error = new ArrayList<>();
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
                    temp = computeOutputValue(anOutput, i);
                    if (temp > max) {
                        max = anOutput;
                        biggestOutNeuron = anOutput;
                    }
                }

                addError(i, neurons.get(biggestOutNeuron).targetValue.get(i) - neurons.get(biggestOutNeuron).input.get(i));
            }

            if (Double.compare(computeMSE(), MSEThreshold) < 0) {
                break;
            }
        }

        printNeuron();
        printWeight();
    }

    public double classifyInstance(Instance instance) {
        List<Integer> hiddNeuron = neuronType.get("hidden");
        List<Integer> outNeuron = neuronType.get("output");
        List<Integer> neuronBefore;
        double weight, maxValue = 0;
        int classIndex = 0;

        for (Integer hidden : hiddNeuron) {
            neurons.put(hidden, new Neuron());
        }
        for (Integer out : outNeuron) {
            neurons.put(out, new Neuron());
        }

        for (Integer aHiddNeuron : hiddNeuron) {

            weight = 0;
            neuronBefore = findNodeBefore(aHiddNeuron);

            for (int j = 0; j < neuronBefore.size(); j++) {
                if (j == 0) { //bias
                    weight += weights.get(aHiddNeuron).get(j)[0];
                } else {
                    weight += weights.get(aHiddNeuron).get(j)[0] * instance.value(j - 1);
                }
            }
            neurons.get(aHiddNeuron).input = new TreeMap<>();
            neurons.get(aHiddNeuron).input.put(0, ActivationClass.sigmoid(weight));
        }

        for (Integer anOutNeuron : outNeuron) {

            weight = 0;
            neuronBefore = findNodeBefore(anOutNeuron);

            for (int j = 0; j < neuronBefore.size(); j++) {
                if (j == 0) { //bias
                    weight += weights.get(anOutNeuron).get(j)[0];
                } else {
                    weight += weights.get(anOutNeuron).get(neuronBefore.get(j))[0] * neurons.get(neuronBefore.get(j)).input.get(0);
                }
            }
            neurons.get(anOutNeuron).input = new TreeMap<>();
            neurons.get(anOutNeuron).input.put(0, ActivationClass.sigmoid(weight));
        }

        for (Integer out : outNeuron) {
            if (Double.compare(neurons.get(out).input.get(0), maxValue) > 0) {
                maxValue = neurons.get(out).input.get(0);
                classIndex = out;
            }
        }

        return classMap.get(classIndex);
    }
}
