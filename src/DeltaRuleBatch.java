import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
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
    public DeltaRuleBatch(Double learningRate,int maxEpoch,Double threshold) {
        super(learningRate,maxEpoch,threshold);
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
        Double newWeight[] = new Double[numAttributes];
        if (isRandom) {
            Random random = new Random();
            for (int i=0;i<numAttributes;i++) {
                newWeight[i] = (double) random.nextInt(1);
            }
        } else {
            for (int i=0;i<numAttributes;i++) {
                newWeight[i] = 1.0;
            }
        }
        inputWeight.add(newWeight);
    }

    @Override
    public void loadTargetFromInstances(Instances instances) {
        int numInstance = instances.numInstances();
        for (int i=0;i<numInstance;i++) {
            target.add(instances.instance(i).classValue());
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
        for (int i=0;i<maxEpoch;i++) {
            
        }
    }

    public static void main(String[] arg) {
        DeltaRule deltaBatchClassifier = new DeltaRuleBatch(0.1,10,0.00001);
        Instances instances = deltaBatchClassifier.readInput("D:\\weka-3-6\\data\\iris.arff");
        try {
            deltaBatchClassifier.buildClassifier(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
