package io.skymind.training.ibm.recurrent.seqClassification;

import java.io.File;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class UCIQuery {

	public static void main(String[] args) throws Exception {
		// Read in the model.
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("target/trained_model.zip"));

		// Training item 0. Correct label is 1.
		double[] seq = new double[] { 24.3566, 36.5163, 38.6025, 46.7903, 46.0926, 34.0541, 33.261, 31.3628, 26.8938,
				12.6914, 12.1574, 16.9906, 23.2936, 22.8489, 35.8733, 41.7359, 47.8879, 38.0542, 44.5395, 37.9045,
				28.2295, 27.0008, 15.9992, 14.1633, 16.5696, 18.0427, 25.5086, 34.0165, 41.38, 45.5368, 45.7775,
				37.9929, 40.2147, 30.7053, 21.7048, 26.0713, 15.6256, 20.1134, 16.9003, 22.4561, 34.1121, 36.8474,
				36.8295, 42.359, 41.7918, 33.1681, 28.0556, 25.5528, 16.2418, 19.1346, 17.0958, 17.8939, 17.4148,
				28.5712, 39.1376, 41.4326, 37.4663, 44.7348, 35.8699, 33.4728 };
		// Create a column vector.
		INDArray testSeq = Nd4j.create(seq, new int[] { seq.length, 1 });
		System.out.println(testSeq);
		System.out.println();

		System.out.println(model.output(testSeq));
		System.out.println("===================");
		System.out.println(model.feedForward(testSeq));
	}

}
