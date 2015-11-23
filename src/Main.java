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
        bp.enumerateClassValue();

        bp.setNominalToBinary();
        bp.setBiasValue(1);
        bp.setBiasWeight(0.1);
        bp.setNumNeuron(2, true); //hidden
        bp.setNumNeuron(bp.data.numClasses(), false); //output
//        bp.setNumOfOutputNeuron();
        bp.setNeuronConnectivity();

        bp.setInstanceTarget(0);
        bp.setInstanceTarget(1);
        bp.setInstanceTarget(2);
        bp.setInstanceTarget(3);


//        System.out.println(bp.findWeight(1,4));
        bp.computeOutputValue(4);
        bp.computeOutputValue(5);
        bp.computeOutputValue(6);
        bp.computeOutputValue(7);
//        bp.computeOutputValue(8);

        /*Map<Integer, Map<Integer, Double>> attribData = bp.parseData();
        for (Map.Entry<Integer, Map<Integer, Double>> data : attribData.entrySet()) {
            System.out.println(data.getValue());
        }*/

        System.out.println("Neuron - Input Value (Activation) - Target - Net Value");
        for (Map.Entry<Integer, Neuron> neuron : bp.neurons.entrySet()) {
            System.out.print(neuron.getKey() + " ");
            System.out.print(neuron.getValue().input);
            System.out.print(" [" + neuron.getValue().targetValue + "] ");
            System.out.println(" [" + neuron.getValue().outValue + "] ");
        }

        System.out.println();
        System.out.println("Connection - New Weight - Old Weight");
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: bp.weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                System.out.print(realWeight.getKey() + "-" + weight.getKey() + " ");
                System.out.print(realWeight.getValue()[0] + " ");
                System.out.println(realWeight.getValue()[1]);
            }
        }

        for (Map.Entry<String, List<Integer>> type : bp.neuronType.entrySet()) {
            System.out.println(type.getKey() + " " + type.getValue());
        }

        /*Map<Integer, Double> neurons = bp.findNodeBefore(7);
        for (Map.Entry<Integer, Double> neuron : neurons.entrySet()) {
            System.out.println(neuron.getKey());
        }*/

        /*Map<Integer, Double> neurons = bp.findNodeAfter(0);
        for (Map.Entry<Integer, Double> neuron : neurons.entrySet()) {
            System.out.println(neuron.getKey());
        }*/
    }
}
