package io.skymind.training.ibm.recurrent.seqClassification;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.berkeley.Pair;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by susaneraly on 11/22/16.
 */
public class UCIData {

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    public static void download() throws Exception{
        if (baseDir.exists()) return;    //Data already exists, don't download it again

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        String[] lines = data.split("\n");

        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));

        int nTrain = 450;   //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
    }
}
