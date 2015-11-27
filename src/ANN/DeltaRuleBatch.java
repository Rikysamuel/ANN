package ANN;

import Util.ActivationClass;
import Util.Util;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class DeltaRuleBatch extends DeltaRule {
    /* the final delta weight per epoch */
    private Double[] finalDeltaWeight;
    /* list final delta weight per epoch */
    private List<Double[]> listFinalDeltaWeight;
    /* the final new weight per epoch */
    private Double[] finalNewWeight;
    /* list final new weight per epoch */
    private List<Double[]> listFinalNewWeight;

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

    /* Default Konstruktor */
    public DeltaRuleBatch() {
        super();
        listFinalDeltaWeight = new ArrayList<>();
        listFinalNewWeight = new ArrayList<>();
    }

    public void setNominalToBinary() {
        NominalToBinary ntb = new NominalToBinary();
        try {
            ntb.setInputFormat(inputDataSet);
            inputDataSet = new Instances(Filter.useFilter(inputDataSet, ntb));
        } catch (Exception e) {
            e.printStackTrace();
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
        for (int j=0;j<numData;j++) {
            Double newWeight[] = new Double[numAttributes-1];
            if (isRandom) {
                Random random = new Random();
                for (int i = 0; i < numAttributes-1; i++) {
                    newWeight[i] = (double) random.nextInt(1);
                }
            } else {
                for (int i = 0; i < numAttributes-1; i++) {
                    newWeight[i] = 0.0;
                }
            }
            inputWeight.add(newWeight);
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
    public void initializeFinalDeltaWeight() {
        finalDeltaWeight = new Double[numAttributes-1];
        for (int i=0;i<numAttributes-1;i++) {
            finalDeltaWeight[i] = 0.0;
        }
    }

    @Override
    public void initializeFinalNewWeight() {
        finalNewWeight = new Double[numAttributes-1];
        for (int i=0;i<numAttributes-1;i++) {
            finalNewWeight[i] = 0.0;
        }
    }

    public Double[] computeSumFinalDeltaWeight() {
        Double[] sumFinalDeltaWeight = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            Double sumDeltaWeightThisAttribute = 0.0;
            for (int j=0;j<numData;j++) {
                sumDeltaWeightThisAttribute += deltaWeight.get(j)[k];
            }
            sumFinalDeltaWeight[k] = sumDeltaWeightThisAttribute;
        }
        return sumFinalDeltaWeight;
    }

    public void initializeInputWeightThisEpoch() {
        for (int k=0;k<numData;k++) {
            Double[] inputWeightThisEpoch = new Double[numAttributes-1];
            for (int i = 0; i < numData; i++) {
                for (int j = 0; j < numAttributes-1; j++) {
                    inputWeightThisEpoch[j] = finalNewWeight[j];
                }
            }
            inputWeight.add(inputWeightThisEpoch);
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
    public double computeOutputInstance(Double[] inputValueThisInstance, Double[] inputWeightThisInstance) {
        double sumNet = 0.0;
        for (int k=0;k<numAttributes-1;k++) {
            sumNet += inputValueThisInstance[k] * inputWeightThisInstance[k];
        }
        return ActivationClass.sigmoid(sumNet);
    }

    @Override
    public Double[] computeDeltaWeightInstance(Double[] inputValueThisInstance, double errorThisInstance, int indexData) {
        Double[] deltaWeightThisInstance = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            double previousDeltaWeightThisAttribute;
            if (indexData > 0) {
                previousDeltaWeightThisAttribute = deltaWeight.get(indexData-1)[k];
            } else {
                previousDeltaWeightThisAttribute = finalDeltaWeight[k];
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
            // Masukkan input weight baru dari epoch sebelumnya
            initializeInputWeightThisEpoch();
            // Proses 1 EPOCH
            for (int j=0;j<numData;j++) {
                // Hitung output data sementara
                double tempOutputThisInstance = computeOutputInstance(inputValue.get(j),inputWeight.get(j));
                // Hitung (target - output) simpan di list sementara
                double tempErrorThisInstance = target.get(j) - tempOutputThisInstance;
                // Hitung deltaweight instance ini di epoch ini
                Double[] deltaWeightThisInstance = computeDeltaWeightInstance(inputValue.get(j),tempErrorThisInstance,j);
                deltaWeight.add(deltaWeightThisInstance);
                // Hitung newweight instance ini di epoch ini
                Double[] newWeightThisInstance = computeNewWeightInstance(inputWeight.get(j),deltaWeightThisInstance);
                newWeight.add(newWeightThisInstance);
            }
            // Calculate sum delta weight this epoch
            finalDeltaWeight =  computeSumFinalDeltaWeight();
            listFinalDeltaWeight.add(computeSumFinalDeltaWeight());
            // Calculate final output for each instance
            finalNewWeight = computeNewWeightInstance(inputWeight.get(numData-1),finalDeltaWeight);
            listFinalNewWeight.add(finalNewWeight);
            for (int j=0;j<numData;j++) {
                Double outputFinalThisData = computeOutputInstance(inputValue.get(j),finalNewWeight);
                output.add(j,outputFinalThisData);
                errorToTarget.add(j,target.get(j)-outputFinalThisData);
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


    public static void main(String[] arg) {
        Util.loadARFF("D:\\weka-3-6\\data\\delta_rule_1.arff");
        Util.buildModel("batch");
    }
}
