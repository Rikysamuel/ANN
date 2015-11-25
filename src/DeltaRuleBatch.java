import weka.core.Instance;
import weka.core.Instances;

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

    /* Default Konstruktor */
    public DeltaRuleBatch() {
        super();
    }

    /* Konstruktor */
    public DeltaRuleBatch(Double learningRate,int maxEpoch,Double threshold,Double momentum) {
        super(learningRate,maxEpoch,threshold,momentum);
    }

    public static Double computeSigmoidFunction(double sumNetFunction) {
        return (1.0 / (1.0 + Math.exp(-1.0 * sumNetFunction)));
    }

    @Override
    public Instances readInput(String filename) {
        FileReader file = null;
        try {
            file = new FileReader(filename);
            try (BufferedReader reader = new BufferedReader(file)) {
                inputDataSet = new Instances(reader);
            }
            inputDataSet.setClassIndex(inputDataSet.numAttributes() - 1);
        } catch (IOException ex) {
            Logger.getLogger(Backpropagation.class.getName()).log(Level.SEVERE, null, ex);
        }
        return inputDataSet;
    }

    @Override
    public void loadInstancesIntoInputValue(Instances instances) {
        numData = instances.numInstances();
        numAttributes = instances.numAttributes();
        for (int i=0;i<numData;i++) {
            Instance thisInstance = instances.instance(i);
            Double[] listInput = new Double[numAttributes];
            for (int j=0;j<numAttributes;j++) {
                listInput[j] = thisInstance.value(j);
            }
            inputValue.add(listInput);
        }
    }

    @Override
    public void loadOrGenerateInputWeight(boolean isRandom) {
        for (int j=0;j<numData;j++) {
            Double newWeight[] = new Double[numAttributes];
            if (isRandom) {
                Random random = new Random();
                for (int i = 0; i < numAttributes; i++) {
                    newWeight[i] = (double) random.nextInt(1);
                }
            } else {
                for (int i = 0; i < numAttributes; i++) {
                    newWeight[i] = 1.0;
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
        finalDeltaWeight = new Double[numAttributes];
        for (int i=0;i<numAttributes;i++) {
            finalDeltaWeight[i] = 0.0;
        }
    }

    @Override
    public double computeDeltaWeight() {

        return 0;
    }

    @Override
    public double computeEpochError() {
        return 0;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        loadInstancesIntoInputValue(instances);
        loadOrGenerateInputWeight(true);
        loadTargetFromInstances(instances);
        initializeFinalDeltaWeight();
        for (int i=0;i<maxEpoch;i++) {
            // List sementara yang menampung error,output,deltaweight,newWeight
            List<Double> tempOutputThisEpoch = new ArrayList<>();
            List<Double> tempErrorThisEpoch = new ArrayList<>();
            List<Double[]> tempDeltaWeightThisEpoch = new ArrayList<>();
            List<Double[]> tempNewWeightThisEpoch = new ArrayList<>();
            // Proses 1 EPOCH
            for (int j=0;j<numData;j++) {
                // Dapatkan list input dan weight di instance dan epoch ini
                Double[] inputValueThisInstance = inputValue.get(j);
                Double[] inputWeightThisInstance = inputWeight.get(j);
                // Hitung output data sementara
                double tempOutputThisInstance;
                double sumNet = 0.0;
                for (int k=0;k<numAttributes;k++) {
                    sumNet += inputValueThisInstance[k] * inputWeightThisInstance[k];
                }
                tempOutputThisInstance = computeSigmoidFunction(sumNet);
                tempOutputThisEpoch.add(tempOutputThisInstance);
                // Hitung (target - output) simpan di list sementara
                double tempErrorThisInstance;
                tempErrorThisInstance = target.get(j) - tempOutputThisInstance;
                tempErrorThisEpoch.add(tempErrorThisInstance);
                // Hitung deltaweight instance ini di epoch ini
                Double[] deltaWeightThisInstance = new Double[numAttributes];
                for (int k=0;k<numAttributes;k++) {
                    deltaWeightThisInstance[k] = learningRate * inputValue.get(j)[k] * tempErrorThisInstance + momentum * finalDeltaWeight[k];
                }
                tempDeltaWeightThisEpoch.add(deltaWeightThisInstance);
                // Hitung newweight instance ini di epoch ini
                Double[] newWeightThisInstance = new Double[numAttributes];
                for (int k=0;k<numAttributes;k++) {
                    newWeightThisInstance[k] = deltaWeightThisInstance[k] + inputWeight.get(j)[k];
                }
                tempNewWeightThisEpoch.add(newWeightThisInstance);
            }
            // Proses pasca 1 EPOCH
            // Hitung Delta Weight final untuk epoch ini
            for (int k=0;k<numAttributes;k++) {
                for (int j=0;j<numData;j++) {
                    finalDeltaWeight[k] += tempDeltaWeightThisEpoch.get(j)[k];
                }
            }
            // Hitung ulang output dan error final untuk query ini
            for (int j=0;j<numData;j++) {
                // Hitung final output per instance untuk epoch ini
                double outputFinalThisInstance;
                double sumNet = 0.0;
                for (int k=0;k<numAttributes;k++) {
                    sumNet += inputValue.get(j)[k] * finalDeltaWeight[k];
                }
                outputFinalThisInstance = computeSigmoidFunction(sumNet);
                output.add(outputFinalThisInstance);
                // Hitung final error per instance untuk epoch ini
                finalErrorToTarget.add(j,target.get(j)-output.get(j));
            }
            // Hitung MSE untuk epoch ini
            double mseValue = 0.0;
            for (int j=0;j<numData;j++) {
                mseValue += 0.5 * Math.pow(finalErrorToTarget.get(j), 2);
            }
            if (mseValue < threshold) {
                isConvergent = true;
            }
        }
    }


    public static void main(String[] arg) {
        DeltaRule deltaBatchClassifier = new DeltaRuleBatch(0.1,10,0.00001,0.1);
        Instances instances = deltaBatchClassifier.readInput("D:\\weka-3-6\\data\\iris.arff");
        try {
            deltaBatchClassifier.buildClassifier(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
