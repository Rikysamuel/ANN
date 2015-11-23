import Util.Neuron;

import java.util.List;
import java.util.Map;

/**
 * Created by rikysamuel on 11/20/2015.
 */
public class Main {
    public static void main(String[] args) {
        BackPropagation bp = new BackPropagation();
        bp.readInput("data/simple.weather.arff");

        bp.setNominalToBinary();
        bp.setBiasValue(1);
        bp.setBiasWeight(0.1);
        bp.setNumNeuron(2, true); //hidden
        bp.setNumNeuron(bp.data.numClasses(), false); //output

        bp.setNeuronConnectivity();

        bp.setInstanceTarget(0);
        bp.setInstanceTarget(1);
        bp.setInstanceTarget(2);
        bp.setInstanceTarget(3);

        bp.computeOutputValue(4);
        bp.computeOutputValue(5);
        bp.computeOutputValue(6);
        bp.computeOutputValue(7);

        bp.computeOutputNeuronError(0);
        bp.computeHiddenNeuronError();
        bp.updateWeights(0);

        System.out.println("Neuron - Input Value (Activation) - Target - Net Value - Error");
        for (Map.Entry<Integer, Neuron> neuron : bp.neurons.entrySet()) {
            System.out.print(neuron.getKey() + " ");
            System.out.print(neuron.getValue().input);
            System.out.print(" "+ neuron.getValue().targetValue + " ");
            System.out.print(" [" + neuron.getValue().outValue + "] ");
            System.out.println(" [" + neuron.getValue().error + "] ");
        }

        System.out.println();
        System.out.println("Connection - New Weight - Old Weight - Delta Weight");
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: bp.weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                System.out.print(realWeight.getKey() + "-" + weight.getKey() + " ");
                System.out.print(realWeight.getValue()[0] + " ");
                System.out.print(realWeight.getValue()[1] + " ");
                System.out.println(realWeight.getValue()[2] + " ");
            }
        }

        System.out.println();
        for (Map.Entry<String, List<Integer>> type : bp.neuronType.entrySet()) {
            System.out.println(type.getKey() + " " + type.getValue());
        }
    }
}
