package ANN;

import Util.ActivationClass;
import Util.Util;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class PerceptronTrainingRule extends Classifier {

    /* the learning rate */
    Double learningRate;
    /* maximum epoch */
    int maxEpoch;
    /* threshold error */
    Double threshold;
    /* momentum */
    Double momentum;
    /* vector of input value for neuron */
    List<Double[]> inputValue;
    /* vector of input weight for neuron */
    List<Double[]> inputWeight;
    /* vector of target value */
    List<Double> target;
    /* the result of net function */
    List<Double> output;

    /* the difference between target and output (t-o) */
    List<Double> errorToTarget;
    /* the delta weight */
    List<Double[]> deltaWeight;
    /* the new weight */
    List<Double[]> newWeight;

    /* the final delta weight per epoch */
    private Double[] finalDeltaWeight;
    /* list final delta weight per epoch */
    private List<Double[]> listFinalDeltaWeight;
    /* the final new weight per epoch */
    private Double[] finalNewWeight;
    /* list final new weight per epoch */
    private List<Double[]> listFinalNewWeight;

    /* dataset input weka */
    Instances inputDataSet;
    /* number of data */
    int numData;
    /* number of attributes */
    int numAttributes;
    /* activation function used */
    String activationFunction;

    /* check if iteration is convergent or not */
    boolean isConvergent;

    public void setNumEpoch(int maxEpoch) {
        this.maxEpoch = maxEpoch;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setThresholdError(double threshold) {
        this.threshold = threshold;
    }

    public void setInputData(Instances inputData) {
        inputDataSet = inputData;
    }

    public void setActivationFunction(String activationFunction){
        this.activationFunction = activationFunction;
    }

    /* Default Constructor */
    public PerceptronTrainingRule() {
        listFinalDeltaWeight = new ArrayList<>();
        listFinalNewWeight = new ArrayList<>();
        inputValue = new ArrayList<>();
        inputWeight = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        errorToTarget = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
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

    /* Initialize final delta weight resulted per epoch */
    public void initializeFinalDeltaWeight() {
        finalDeltaWeight = new Double[numAttributes];
        for (int i=0;i<numAttributes;i++) {
            finalDeltaWeight[i] = 0.0;
        }
    }

    /* Initialize final new weight resulted per epoch */
    public void initializeFinalNewWeight() {
        finalNewWeight = new Double[numAttributes];
        for (int i=0;i<numAttributes;i++) {
            finalNewWeight[i] = 0.0;
        }
    }

    /* Convert instances data into input value */
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

    /* Randomize or generalize weight each input */
    public void loadOrGenerateInputWeight(boolean isRandom) {
        Double newWeight[] = new Double[numAttributes];
        if (isRandom) {
            Random random = new Random();
            for (int i = 0; i < numAttributes; i++) {
                newWeight[i] = (double) random.nextInt(1);
            }
        } else {
            for (int i = 0; i < numAttributes; i++) {
                newWeight[i] = 0.0;
            }
        }
        inputWeight.add(0, newWeight);
    }

    /* Load target value from arff file */
    public void loadTargetFromInstances(Instances instances) {
        int numInstance = instances.numInstances();
        for (int i=0;i<numInstance;i++) {
            target.add(instances.instance(i).classValue());
        }
    }

    /* compute the error of Epoch */
    public double computeEpochError(List<Double> finalErrorThisEpoch) {
        double mseValue = 0.0;
        for (int j=0;j<numData;j++) {
            mseValue += 0.5 * Math.pow(finalErrorThisEpoch.get(j), 2);
        }
        return mseValue;
    }

    /* compute output of one instance using sigmoid activation function */
    public double computeOutputInstance(Double[] inputValueThisInstance, Double[] inputWeightThisInstance) {
        double sumNet = 0.0;
        for (int k=0;k<numAttributes;k++) {
            sumNet += inputValueThisInstance[k] * inputWeightThisInstance[k];
        }

        if(activationFunction.equals("sign")){
            return ActivationClass.sign(sumNet);
        } else { // sigmoid
            return ActivationClass.sigmoid(sumNet);
        }
    }

    /* compute delta weight of one instance */
    public Double[] computeDeltaWeightInstance(Double[] inputValueThisInstance, double errorThisInstance, int indexData) {
        Double[] deltaWeightThisInstance = new Double[numAttributes];
        for (int k=0;k<numAttributes;k++) {
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

    /* compute new weight yields for one instance */
    public Double[] computeNewWeightInstance(Double[] inputWeightThisInstance, Double[] deltaWeightThisInstance) {
        Double[] newWeightThisInstance = new Double[numAttributes];
        for (int k=0;k<numAttributes;k++) {
            newWeightThisInstance[k] = deltaWeightThisInstance[k] + inputWeightThisInstance[k];
        }
        return newWeightThisInstance;
    }

    public void initializeInputWeightThisIteration(int indexData) {
        Double[] inputWeightThisIteration = new Double[numAttributes];
        for (int j = 0; j < numAttributes; j++) {
            inputWeightThisIteration[j] = finalNewWeight[j];
        }
        inputWeight.add(indexData,inputWeightThisIteration);
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
            for (int j=0;j<numData;j++) {
                // Inisialisasi input weight baru untuk iterasi ini
                initializeInputWeightThisIteration(j);
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
                // Masukkan final delta weight dan new weight untuk iterasi ini
                finalDeltaWeight = deltaWeightThisInstance;
                finalNewWeight = newWeightThisInstance;
            }
            // Add last final new weight this epoch into list
            listFinalDeltaWeight.add(finalDeltaWeight);
            // Calculate final output for each instance
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
            }
        }
    }

    public static void main(String[] arg) {
        Util.loadARFF("C:\\Program Files (x86)\\Weka-3-7\\data\\weather.numeric.arff");
        Util.buildModel("perceptron");
    }

}
