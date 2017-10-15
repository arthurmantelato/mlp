package br.com.arthurmantelato.ia.data;

import org.assertj.core.api.Assertions;
import org.junit.Test;

public class TimeSerieStaticTransformerTest {

	@Test
	public void testTransform6to2by2() {
		TimeSerieStaticTransformer transformer = new TimeSerieStaticTransformer();
		String datasetFilePath = "./src/test/resources/timeseries.txt";
		int outputDimension = 2;
		double[][] actual = transformer.transform(datasetFilePath, outputDimension);
		double[][] expected = { { 1.0, 2.0, 3.0 }, { 2.0, 3.0, 4.0 }, { 3.0, 4.0, 5.0 }, { 4.0, 5.0, 6.0 } };
		Assertions.assertThat(actual).isEqualTo(expected);
	}
	
	@Test
	public void testTransform6to4by4() {
		TimeSerieStaticTransformer transformer = new TimeSerieStaticTransformer();
		String datasetFilePath = "./src/test/resources/timeseries.txt";
		int outputDimension = 4;
		double[][] actual = transformer.transform(datasetFilePath, outputDimension);
		double[][] expected = { { 1.0, 2.0, 3.0 , 4.0, 5.0}, { 2.0, 3.0, 4.0, 5.0, 6.0 } };
		Assertions.assertThat(actual).isEqualTo(expected);
	}

}
