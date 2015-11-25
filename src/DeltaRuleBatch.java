import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by rikysamuel on 11/4/2015.
 */
public class DeltaRuleBatch extends DeltaRule {
    /* dataset input weka */
    Instances inputDataSet;

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
        int numInstance = instances.numInstances();
        for (int i=0;i<numInstance;i++) {
            Instance thisInstance = instances.instance(i);
            Double[] listInput = new Double[thisInstance.numAttributes()];
            
        }
    }

    @Override
    public void loadOrGenerateInputWeight(boolean isRandom, Double[] weight) {

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

    }
}
