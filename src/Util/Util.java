package Util;

import ANN.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by rikysamuel on 11/26/2015.
 */
public class Util {
    private static Classifier classifier;
    private static Instances data;

    public static Instances getData() {
        return data;
    }

    public static Classifier getClassifier() {
        return classifier;
    }

    /**
     * load dataset from ARFF format
     * @param filename file path
     */
    public static void loadARFF(String filename) {
        FileReader file = null;
        try {
            file = new FileReader(filename);
            try (BufferedReader reader = new BufferedReader(file)) {
                data = new Instances(reader);
            }
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
        } catch (IOException ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (file!=null) {
                    file.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    /**
     * load dataset from CSV format
     * @param filename
     */
    public static void loadCSV(String filename) {
        try {
            CSVLoader csv = new CSVLoader();
            csv.setFile(new File(filename));
            data = csv.getDataSet();

            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
        } catch (IOException ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * remove a certain attribute from the dataset
     * @param id attribute id to remove
     */
    public static void removeAttribute(int id) {
        data.deleteAttributeAt(id);
    }

    /**
     * resample the instances in dataset
     * @param seed random seed
     * @return resampled instances
     */
    public static Instances resample(int seed) {
        return data.resample(new Random(seed));
    }

    public static Instances setNominalToBinary(Instances instances) {
        NominalToBinary ntb = new NominalToBinary();
        Instances newInstances = null;
        try {
            ntb.setInputFormat(instances);
            newInstances = new Instances(Filter.useFilter(instances, ntb));
        } catch (Exception e) {
            e.printStackTrace();
        }

        return newInstances;
    }

    public static Instances useNormalization(Instances instances) {
        Normalize normalize = new Normalize();
        Instances newInstances = null;
        try {
            normalize.setInputFormat(instances);
            newInstances = new Instances(Filter.useFilter(instances, normalize));
        } catch (Exception e) {
            e.printStackTrace();
        }

        return newInstances;
    }

    /**
     * apply all filter to build the classifier
     * @param Classifier model
     */
    public static void buildModel(String Classifier) {
        try {
            // Membangun model dan melakukan test
            switch (Classifier.toLowerCase()) {
                case "mlp" :
                    BackPropagation bp = new BackPropagation();
                    data = Util.setNominalToBinary(data);
                    data = Util.useNormalization(data);
                    bp.data = data;
                    bp.setNumOfInputNeuron();
                    bp.setBiasValue(1);
                    bp.setBiasWeight(0.1);
                    bp.setInitWeight(0.1);
                    bp.setNumNeuron(2, true); //hidden
                    bp.setMomentum(0.1);
                    bp.setLearningRate(0.1);
                    bp.setNumEpoch(1000);

                    classifier = bp;
                    break;
                case "batch" :
                    DeltaRuleBatch batch = new DeltaRuleBatch();
                    batch.setInputData(data);
                    batch.setNominalToBinary();
                    batch.setLearningRate(0.1);
                    batch.setMomentum(0.1);
                    batch.setNumEpoch(10);
                    batch.setThresholdError(0.001);

                    classifier = batch;
                    break;
                case "incremental" :
                    DeltaRuleIncremental incremental = new DeltaRuleIncremental();
                    incremental.setInputData(data);
                    incremental.setNominalToBinary();
                    incremental.setLearningRate(0.1);
                    incremental.setMomentum(0.1);
                    incremental.setNumEpoch(10);
                    incremental.setThresholdError(0.001);

                    classifier = incremental;
                    break;
                case "perceptron" :
                    break;
                default :
                    break;
            }
            classifier.buildClassifier(data);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * save classifier to a specific location
     * @param filePath  filepath name
     */
    public static void saveClassifier(String filePath) {
        try {
            SerializationHelper.write(filePath, classifier);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * load classifier from .model file
     * @param modelPath model file path
     */
    public static void loadClassifer(String modelPath) {
        try {
            classifier = (Classifier) SerializationHelper.read(modelPath);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * show learning statistic result by folds-cross-validation
     * @param data instances
     * @param folds num of folds
     */
    public static void FoldSchema(Instances data, int folds) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(Util.getClassifier(), data, folds, new Random(1));
            System.out.println(eval.toSummaryString("\nResults " + folds + " folds cross-validation\n\n", false));
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * show learning statistic result by full-training method
     * @param data training data
     */
    public static void FullSchema(Instances data) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(classifier, data);
            System.out.println(eval.toSummaryString("\nResults Full-Training\n\n", false));
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * show learning statistic result by using test sets
     * @param testPath test path file
     * @param typeTestFile test file
     */
    public static void TestSchema(String testPath, String typeTestFile) {
        Instances testsets = null;
        // Load test instances based on file type and path
        if (typeTestFile.equals("arff")) {
            FileReader file = null;
            try {
                file = new FileReader(testPath);
                try (BufferedReader reader = new BufferedReader(file)) {
                    testsets = new Instances(reader);
                }
                // setting class attribute
                testsets.setClassIndex(data.numAttributes() - 1);
            } catch (IOException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            } finally {
                try {
                    if (file!=null) {
                        file.close();
                    }
                } catch (IOException ex) {
                    Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        } else if (typeTestFile.equals("csv")) {
            try {
                CSVLoader csv = new CSVLoader();
                csv.setFile(new File(testPath));
                data = csv.getDataSet();

                // setting class attribute
                data.setClassIndex(data.numAttributes() - 1);
            } catch (IOException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        // Start evaluate model using instances test and print results
        try {
            Evaluation eval = new Evaluation(Util.getData());
            eval.evaluateModel(Util.getClassifier(), testsets);
            System.out.println(eval.toSummaryString("\nResults\n\n", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * show learning statistic result by percentage split
     * @param data training data
     * @param attributeIndices attribute number to train
     * @param trainPercent percentage of the training data
     * @param Classifier model
     */
    public static void PercentageSplit(Instances data, String attributeIndices, double trainPercent, String Classifier) {
        try {
            int trainSize = (int) Math.round(data.numInstances()* trainPercent / 100);
            int testSize = data.numInstances() - trainSize;

            data.randomize(new Random(1));

            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            switch (Classifier.toLowerCase()) {
                case "mlp" :
                    classifier = new BackPropagation();
                    break;
                case "batch" :
                    classifier = new DeltaRuleBatch();
                    break;
                case "incremental" :
                    classifier = new DeltaRuleIncremental();
                    break;
                case "perceptron" :
                    classifier = new PerceptronTrainingRule();
                    break;
                default :
                    break;
            }
            classifier.buildClassifier(train);

            for (int i = 0; i < test.numInstances(); i++) {
                try {
                    double pred = classifier.classifyInstance(test.instance(i));
                    System.out.print("ID: " + test.instance(i));
                    System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
                    System.out.println(", predicted: " + test.classAttribute().value((int) pred));
                } catch (Exception ex) {
                    Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
                }
            }

            // Start evaluate model using instances test and print results
            try {
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } catch (Exception e) {
                e.printStackTrace();
            }

        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    /**
     * Classify test set using pre-build model
     * @param model model pathfile
     * @param test test file
     */
    public static void doClassify(Classifier model, Instances test) {
        test.setClassIndex(test.numAttributes() - 1);
        for (int i = 0; i < test.numInstances(); i++) {
            try {
                double pred = model.classifyInstance(test.instance(i));
                System.out.print("ID: " + test.instance(i));
                System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
                System.out.println(", predicted: " + test.classAttribute().value((int) pred));
            } catch (Exception ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
}
