package ANN;

import Util.ActivationClass;
import Util.Util;
import com.sun.jmx.snmp.Enumerated;
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
 * Created by rikysamuel on 11/4/2015.
 */
public class DeltaRuleIncremental extends DeltaRule {
    /* the final delta weight per epoch */
    private List<Double[]> finalDeltaWeight;
    /* the final new weight per epoch */
    private List<Double[]> finalNewWeight;

    public void setNumEpoch(int maxEpoch) {
        super.maxEpoch = maxEpoch;
    }

    public void setLearningRate(double learningRate) {
        super.learningRate = learningRate;
    }

    public void setMomentum(double momentum) {
        super.momentum = momentum;
    }

    public void setThresholdError(double threshold) {
        super.threshold = threshold;
    }

    public void setInputData(Instances inputData) {
        super.inputDataSet = inputData;
    }

    /* Default Constructor */
    public DeltaRuleIncremental() {
        super();
        finalDeltaWeight = new ArrayList<>();
        finalNewWeight = new ArrayList<>();
    }

    @Override
    public void initializeFinalDeltaWeight() {
        finalDeltaWeight.clear();
        for (int i=0;i<numClasses;i++) {
            Double[] finalDeltaWeightPerClass = new Double[numAttributes-1];
            for (int j=0;j<numAttributes-1;j++) {
                finalDeltaWeightPerClass[j] = 0.0;
            }
            finalDeltaWeight.add(finalDeltaWeightPerClass);
        }
    }

    @Override
    public void initializeFinalNewWeight() {
        finalNewWeight.clear();
        for (int i=0;i<numClasses;i++) {
            Double[] finalNewWeightPerClass = new Double[numAttributes-1];
            for (int j=0;j<numAttributes-1;j++) {
                finalNewWeightPerClass[j] = 0.0;
            }
            finalNewWeight.add(finalNewWeightPerClass);
        }
    }

    @Override
    public void loadInstancesIntoInputValue(Instances instances) {
        numData = instances.numInstances();
        numAttributes = instances.numAttributes();
        for (int i=0;i<numData;i++) {
            Instance thisInstance = instances.instance(i);
            Double[] listInput = new Double[numAttributes-1];
            for (int j=0;j<numAttributes-1;j++) {
                listInput[j] = thisInstance.value(j);
            }
            inputValue.add(listInput);
        }
    }

    @Override
    public void loadOrGenerateInputWeight(boolean isRandom) {
        for (int i=0;i<numClasses;i++) {
            List<Double[]> listInputWeightPerClass = new ArrayList<>();
            for (int j=0;j<numData;j++) {
                Double[] inputWeightPerData = new Double[numAttributes-1];
                for (int k=0;k<numAttributes-1;k++) {
                    if (isRandom) {
                        Random random = new Random();
                        inputWeightPerData[k] = (double) random.nextInt(1);
                    } else {
                        inputWeightPerData[k] = 0.0;
                    }
                }
                listInputWeightPerClass.add(inputWeightPerData);
            }
            inputWeight.add(listInputWeightPerClass);
        }
    }

    @Override
    public void loadTargetFromInstances(Instances instances) {
        int numInstance = instances.numInstances();
        numClasses = instances.numClasses();
        for (int i=0;i<numClasses;i++) {
            List<Double> listTargetThisClass = new ArrayList<>();
            for (int j=0;j<numInstance;j++) {
                if (instances.instance(j).classValue() == (double) i) {
                    listTargetThisClass.add(1.0);
                } else {
                    listTargetThisClass.add(0.0);
                }
            }
            target.add(listTargetThisClass);
        }
    }

    @Override
    public double computeEpochError(List<Double> finalErrorThisEpoch) {
        double mseValue = 0.0;
        for (int j=0;j<numData;j++) {
            mseValue += 0.5 * Math.pow(finalErrorThisEpoch.get(j), 2);
        }
        return mseValue;
    }

    @Override
    public Double computeOutputInstance(Double[] inputValueThisInstance, Double[] inputWeightThisInstance) {
        double sumNet = 0.0;
        for (int k=0;k<numAttributes-1;k++) {
            sumNet += inputValueThisInstance[k] * inputWeightThisInstance[k];
        }
        return sumNet;
    }

    @Override
    public Double[] computeDeltaWeightInstance(Double[] inputValueThisInstance, Double errorThisInstance, int indexData, int neuronOutputIndex) {
        Double[] deltaWeightThisInstance = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            double previousDeltaWeightThisAttribute;
            if (indexData > 0) {
                previousDeltaWeightThisAttribute = deltaWeight.get(neuronOutputIndex).get(indexData-1)[k];
            } else {
                previousDeltaWeightThisAttribute = finalDeltaWeight.get(neuronOutputIndex)[k];
            }
            deltaWeightThisInstance[k] = learningRate * inputValueThisInstance[k] * errorThisInstance + momentum * previousDeltaWeightThisAttribute;
        }
        return deltaWeightThisInstance;
    }

    @Override
    public Double[] computeNewWeightInstance(Double[] inputWeightThisInstance, Double[] deltaWeightThisInstance) {
        Double[] newWeightThisInstance = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            newWeightThisInstance[k] = deltaWeightThisInstance[k] + inputWeightThisInstance[k];
        }
        return newWeightThisInstance;
    }

    @Override
    public Double computeErrorThisInstance(Double targetOutputPerNeuron, Double outputPerNeuron) {
        return (targetOutputPerNeuron-outputPerNeuron);
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

    public void initializeInputWeightThisIteration(int outputNeuronIndex) {
        List<Double[]> listInputWeightPerClass = new ArrayList<>();
        for (int j=0;j<numData;j++) {
            Double[] inputWeightPerIteration = new Double[numAttributes-1];
            for (int k=0;k<numAttributes-1;k++) {
                inputWeightPerIteration[k] = finalNewWeight.get(outputNeuronIndex)[k];
            }
            listInputWeightPerClass.add(inputWeightPerIteration);
        }
        inputWeight.add(outputNeuronIndex,listInputWeightPerClass);
    }

    public void resetDataPerEpoch() {
        // Reset isi dari delta weight dan new weight
        output.clear();
        errorToTarget.clear();
        inputWeight.clear();
        deltaWeight.clear();
        newWeight.clear();
        for (int j=0;j<numClasses;j++) {
            deltaWeight.add(new ArrayList<>());
            newWeight.add(new ArrayList<>());
        }
    }

    public int indexClassWithHighestOutput(List<Double> outputEachNeuron) {
        int indexClass = 0;
        Double maxOutput = outputEachNeuron.get(0);
        for (int i=1;i<outputEachNeuron.size();i++) {
            if (outputEachNeuron.get(i) > maxOutput) {
                indexClass = i;
                maxOutput = outputEachNeuron.get(i);
            }
        }
        return indexClass;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        loadInstancesIntoInputValue(instances);
        loadTargetFromInstances(instances);
        loadOrGenerateInputWeight(false);
        initializeFinalDeltaWeight();
        initializeFinalNewWeight();
        for (int i=0;i<maxEpoch;i++) {
            // Reset isi dari delta weight dan new weight
            resetDataPerEpoch();
            // Proses 1 EPOCH
            for (int j=0;j<numData;j++) {
                for (int k=0;k<numClasses;k++) {
                    // Masukkan input weight dari iterasi sebelumnya
                    initializeInputWeightThisIteration(k);
                    // Hitung output data sementara
                    double tempOutputThisInstance = computeOutputInstance(inputValue.get(j), inputWeight.get(k).get(j));
                    // Hitung (target - output) sementara
                    double tempErrorThisInstance = computeErrorThisInstance(target.get(k).get(j), tempOutputThisInstance);
                    // Hitung deltaweight instance ini di epoch ini
                    Double[] deltaWeightThisInstance = computeDeltaWeightInstance(inputValue.get(j),tempErrorThisInstance,j,k);
                    deltaWeight.get(k).add(j, deltaWeightThisInstance);
                    finalDeltaWeight.set(k, deltaWeightThisInstance);
                    // Hitung newweight instance ini di epoch ini
                    Double[] newWeightThisInstance = computeNewWeightInstance(inputWeight.get(k).get(j), deltaWeightThisInstance);
                    newWeight.get(k).add(j, newWeightThisInstance);
                    finalNewWeight.set(k, newWeightThisInstance);
                }
            }
            // Isi error to target akhir sebelum menghitung MSE
            for (int j=0;j<numData;j++) {
                List<Double> listOutputThisInstance = new ArrayList<>();
                for (int k=0;k<numClasses;k++) {
                    Double outputFinalThisClass = computeOutputInstance(inputValue.get(j),finalNewWeight.get(k));
                    listOutputThisInstance.add(outputFinalThisClass);
                }
                int indexClassWithHighestOutput = indexClassWithHighestOutput(listOutputThisInstance);
                Collections.sort(listOutputThisInstance);
                Double finalOutputThisInstance = listOutputThisInstance.get(numClasses-1);
                Double finalErrorThisInstance = computeErrorThisInstance(target.get(indexClassWithHighestOutput).get(j), finalOutputThisInstance);
                errorToTarget.add(finalErrorThisInstance);
            }
            // Hitung MSE Error epoch ini
            double mseValue = computeEpochError(errorToTarget);
            System.out.println("Error epoch " + (i+1) + " : " + mseValue);
            if (mseValue < threshold) {
                isConvergent = true;
                break;
            }
        }
    }

    public double classifyInstance(Instance instance) {
        // Masukkan input value tiap attribute pada instance
        Double[] inputValue = new Double[numAttributes-1];
        for (int i=0;i<instance.numAttributes()-1;i++) {
            inputValue[i] = instance.value(i);
        }
        // Hitung output setiap neuron, cari yang terbesar
        List<Double> outputEachNeuron = new ArrayList<>();
        for (Double[] newWeight : finalNewWeight) {
            Double outputThisNeuron = computeOutputInstance(inputValue,newWeight);
            outputEachNeuron.add(outputThisNeuron);
        }
        return indexClassWithHighestOutput(outputEachNeuron);
    }

    public static void main(String[] arg) {
        Util.loadARFF("D:\\weka-3-6\\data\\weather.nominal.arff");
        Util.buildModel("incremental");
       // Util.FoldSchema(Util.getData(),10);
       /* Enumeration inst = Util.getData().enumerateInstances();
        while (inst.hasMoreElements()) {
            Instance instance = (Instance) inst.nextElement();
            try {
                System.out.println(Util.getClassifier().classifyInstance(instance));
            } catch (Exception e) {
                e.printStackTrace();
            }
        } */
    }
}
