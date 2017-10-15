package br.com.arthurmantelato.ia.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class TimeSerieStaticTransformer {

	public double[][] transform(String datasetFilePath, int outputDimension) {
		try (BufferedReader reader = new BufferedReader(new FileReader(datasetFilePath))) {
			double[] inputs = reader.lines().mapToDouble(s -> Double.parseDouble(s)).toArray();
			int currentInputIndex = 0;
			int outputCount = inputs.length / outputDimension + 1;
			double[][] output = new double[outputCount][outputDimension + 1];
			for (int i = 0; i < outputCount; i++) {
				double[] transformedInput = new double[outputDimension + 1];
				for (int d = 0; d < outputDimension + 1 && d < outputCount && currentInputIndex < inputs.length; d++) {
					transformedInput[d] = inputs[currentInputIndex++];
				}
				currentInputIndex = currentInputIndex - outputDimension;
				output[i] = transformedInput;
			}
			return output;
		} catch (IOException e) {
			e.printStackTrace();
		}
		;

		return null;
	}

}
