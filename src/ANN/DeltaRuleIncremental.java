package ANN;

import Util.ActivationClass;
import Util.Util;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
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
        for (int i=0;i<numInstance;i++) {
            target.add(instances.instance(i).classValue());
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

    public void initializeInputWeightThisIteration(int indexData, int outputNeuronIndex) {
        Double[] inputWeightThisIteration = new Double[numAttributes-1];
        for (int j = 0; j < numAttributes-1; j++) {
            inputWeightThisIteration[j] = finalNewWeight.get(outputNeuronIndex)[j];
        }
        inputWeight.add(new ArrayList<>());
        inputWeight.get(outputNeuronIndex).add(indexData, inputWeightThisIteration);
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
            deltaWeight.clear();
            newWeight.clear();
            inputWeight.clear();
            output.clear();
            errorToTarget.clear();
            // Proses 1 EPOCH
            for (int j=0;j<numClasses;j++) {
                List<Double[]> listInputWeightThisClass = inputWeight.get(j);
                deltaWeight.add(new ArrayList<>());
                newWeight.add(new ArrayList<>());
                for (int k=0;k<numData;k++) {
                    initializeInputWeightThisIteration(k,j);
                    // Hitung output data sementara
                    double tempOutputThisInstance = computeOutputInstance(inputValue.get(k),listInputWeightThisClass.get(k));
                    // Hitung (target - output) sementara
                    double tempErrorThisInstance = computeErrorThisInstance(target.get(k),tempOutputThisInstance);
                    // Hitung deltaweight instance ini di epoch ini
                    Double[] deltaWeightThisInstance = computeDeltaWeightInstance(inputValue.get(k),tempErrorThisInstance,k,j);
                    deltaWeight.get(j).add(deltaWeightThisInstance);
                    finalDeltaWeight.add(deltaWeightThisInstance);
                    // Hitung newweight instance ini di epoch ini
                    Double[] newWeightThisInstance = computeNewWeightInstance(listInputWeightThisClass.get(k), deltaWeightThisInstance);
                    newWeight.get(j).add(newWeightThisInstance);
                    finalNewWeight.add(newWeightThisInstance);
                }
            }
            // Isi error to target akhir sebelum menghitung MSE
            for (int j=0;j<numClasses;j++) {
                Double[] finalWeightThisClass = finalNewWeight.get(j);
                output.add(new ArrayList<>());
                for (int k=0;k<numData;k++) {
                    output.get(j).add(computeOutputInstance(inputValue.get(k), finalWeightThisClass));
                }
                Collections.sort(output.get(j));
                errorToTarget.add(output.get(j).get(output.get(j).size() - 1));
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
        return 0;
    }

    public static void main(String[] arg) {
        Util.loadARFF("D:\\weka-3-6\\data\\delta_rule_1.arff");
        Util.buildModel("incremental");
    }
}
