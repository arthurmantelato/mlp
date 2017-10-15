package br.com.arthurmantelato.ia.mlp;

import java.io.File;
import java.util.function.Function;

import Jama.Matrix;
import br.com.arthurmantelato.ia.data.TimeSerieStaticTransformer;

public class MultiLayerPerceptron {

	private static final double BIAS_VALUE = 1.0;
	private static final double BIAS_WEIGHT = 0.0;
	
	private static final int HIDDEN_LAYER_WEIGHTS_INDEX = 0;
	private static final int OUTPUT_LAYER_WEIGHTS_INDEX = 1;

	public static void main(String[] args) {
		
		String datasetRootDirPath = "/home/fieldy/Dropbox/USP/Mestrado/inteligencia-computacional/Trabalhos/RedeNeuralRecorrente";
		String datasetFile = "serie3_trein.txt";
		
		String dataSetFilePath = datasetRootDirPath + File.separator + datasetFile;
		
		int inputDataDimension = 2;
		TimeSerieStaticTransformer timeSerieStaticTransformer = new TimeSerieStaticTransformer();
		double[][] inputData = timeSerieStaticTransformer.transform(dataSetFilePath, inputDataDimension);
		Matrix input = new Matrix(inputData);
		
		int hiddenLayerNeuronsAmount = 4;
		int outputLayerNeuronsAmount = 1;
		Matrix[] weights = { Matrix.random(hiddenLayerNeuronsAmount, inputDataDimension),
				Matrix.random(outputLayerNeuronsAmount, hiddenLayerNeuronsAmount) };

		
		double maxError = 1e-3;
		double error = Double.MAX_VALUE;
		int maxEpoches = 5000;
		double learningRate = 0.3;
		
		int epoch = 0;
		MultiLayerPerceptron mlp = new MultiLayerPerceptron();
		while(error > maxError && epoch < maxEpoches) {
			double epochError = 0.0;
			int currentInput = 0;
			for(int i = 0; i < input.getRowDimension(); i++) {
				Matrix inputDataMatrix = input.getMatrix(i, currentInput++, 0, input.getColumnDimension() -2);
				Matrix[] output = mlp.predict(inputDataMatrix , weights);
				double expected = input.get(i, inputDataDimension - 1);
				double predicted = output[output.length - 1].get(0, 0);
				double predictionError = expected - predicted;
				mlp.updateWeights(weights, predictionError, output);
				System.out.println(predictionError);
				epochError = epochError + predictionError;
			}
			error = 0.5 * epochError * epochError;
			epoch++;
			System.out.println("EPOCH="+epoch+", error="+error);
		}
		
	}

	private void updateWeights(Matrix[] weights, double predictionError, Matrix[] output) {
		for(int i = 0; i < weights[OUTPUT_LAYER_WEIGHTS_INDEX].getRowDimension(); i++) {
			Matrix matrix = output[3];
			double newWeight = predictionError * sigmoidDerivate.apply(matrix.get(i, 0)) * output[2].get(i, 0);
			weights[OUTPUT_LAYER_WEIGHTS_INDEX].set(i, 0, newWeight);
		}
		
		for(int i = 0; i < weights[HIDDEN_LAYER_WEIGHTS_INDEX].getRowDimension(); i++) {
			double newWeight = 0.0;
			weights[HIDDEN_LAYER_WEIGHTS_INDEX].set(i, 0, newWeight);
		}
	}

	private Matrix[] predict(Matrix inputData, Matrix[] weights) {
		
		Matrix X = new Matrix(1, inputData.getColumnDimension() + 1);
		X.setMatrix(0, inputData.getRowDimension() - 1, 0, inputData.getColumnDimension() - 1, inputData);
		X.set(0, inputData.getColumnDimension(), BIAS_VALUE);
		
		Matrix A = appendColumnWithValue(weights[HIDDEN_LAYER_WEIGHTS_INDEX], BIAS_WEIGHT);
		
		// Z_in(n) = sum(A*X(n)')
		Matrix Z_in = X.times(A.transpose());
		
		// Z(n) = f(Z_in(n))
		Matrix Z = applyFunction(Z_in, Math::tanh);

		Matrix temp = new Matrix(1, Z.getColumnDimension() + 1);
		temp.setMatrix(0, Z.getRowDimension() - 1, 0, Z.getColumnDimension() -1, Z);
		temp.set(0, Z.getColumnDimension(), BIAS_VALUE);
		Z = temp;
		
		Matrix B = appendColumnWithValue(weights[OUTPUT_LAYER_WEIGHTS_INDEX], BIAS_WEIGHT);
		
		Matrix Y_in = Z.times(B.transpose());
		
		// Y(n) = g(Y_in(n))
		Matrix Y = applyFunction(Y_in, sigmoid);
		
		return new Matrix[] {X, Z_in, Z, Y_in, Y};
	}

	private void printMatrix(String matrixName, Matrix a) {
		//System.out.println(matrixName);
		//a.print(15, 12);
	}

	private Matrix applyFunction(Matrix input, Function<Double, Double> activationFunction) {
		double[][] outputValues = input.getArrayCopy();
		for (int i = 0; i < input.getRowDimension(); i++) {
			for (int j = 0; j < input.getColumnDimension(); j++) {
				outputValues[i][j] = activationFunction.apply(input.get(i, j));
			}
		}
		return new Matrix(outputValues);
	}

	private Matrix appendColumnWithValue(Matrix matrix, double value) {
		int r = matrix.getRowDimension();
		int c = matrix.getColumnDimension();

		Matrix columnToAppend = new Matrix(r, 1, value);

		Matrix result = new Matrix(r, c + 1);
		result.setMatrix(0, r - 1, 0, c - 1, matrix);
		result.setMatrix(0, r - 1, new int[] { c }, columnToAppend);

		return result;
	}
	
	private Function<Double, Double> hiperbolicTangent = x -> (Math.exp(x) - Math.exp(-x))/(Math.exp(x) + Math.exp(-x));
	
	private Function<Double, Double> hiperbolicTangentDerivate = x -> 1 - Math.pow(hiperbolicTangent.apply(x), 2);
	
	private Function<Double, Double> sigmoid = x -> 1/(1 - Math.exp(x));
	
	private Function<Double, Double> sigmoidDerivate = x -> (1 - sigmoid.apply(x))*sigmoid.apply(x);

};