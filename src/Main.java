import Util.Neuron;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Main {

    public void printNeuron(Map<Integer, Neuron> neurons) {
        System.out.println("Neuron - Input Value (Activation) - Target - Net Value - Error");
        for (Map.Entry<Integer, Neuron> neuron : neurons.entrySet()) {
            System.out.print(neuron.getKey() + " ");
            System.out.print(neuron.getValue().input);
            System.out.print(" "+ neuron.getValue().targetValue + " ");
            System.out.print(" [" + neuron.getValue().outValue + "] ");
            System.out.println(" [" + neuron.getValue().error + "] ");
        }
    }

    public void printWeight(Map<Integer, Map<Integer, Double[]>> weights) {
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

    public static void main(String[] args) {
        BackPropagation bp = new BackPropagation();
        bp.readInput("data/simple.weather.arff");

        /* config */
        bp.setNominalToBinary();
        bp.setBiasValue(1);
        bp.setBiasWeight(0.1);
        bp.setNumNeuron(2, true); //hidden
        bp.setNumNeuron(bp.data.numClasses(), false); //output
        bp.setNeuronConnectivity();

        List<Integer> hidden = bp.neuronType.get("hidden");
        List<Integer> output = bp.neuronType.get("output");

        for (int epoch = 0; epoch < 2; epoch++) {
            for (int i = 0; i < bp.data.numInstances(); i++) {
                System.out.println("instance num: " + i);
                bp.setInstanceTarget(i);

                for (int j = 0; j < hidden.size(); j++) {
                    bp.computeOutputValue(hidden.get(j), i);
                }
                for (int j = 0; j < output.size(); j++) {
                    bp.computeOutputValue(output.get(j), i);
                }

                bp.computeOutputNeuronError(i);
                bp.computeHiddenNeuronError(i);

                bp.updateWeights(i);

                for (int j = 0; j < hidden.size(); j++) {
                    bp.computeOutputValue(hidden.get(j), i);
                }
                for (int j = 0; j < output.size(); j++) {
                    bp.computeOutputValue(output.get(j), i);
                }

                new Main().printNeuron(bp.neurons);
                new Main().printWeight(bp.weights);
            }
        }
    }
}
