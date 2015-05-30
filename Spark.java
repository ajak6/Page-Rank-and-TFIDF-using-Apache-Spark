import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class Spark {
  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws IOException {
    SparkConf sparkConf = new SparkConf().setAppName("Step 1 tfidf");
    JavaSparkContext spark = new JavaSparkContext(sparkConf);

    // JavaRDD<String> file =
    JavaRDD<String> file = spark.textFile("s3n://s15-p42-part1-easy/data/");
    // from the file take out the 1st vaue as document name
    // from the file take out the 4th field after \t as text for which words
    // are to be counted
    // step 1: compute TE=word count per file
    JavaRDD<String> wordTitle = file
        .flatMap(new FlatMapFunction<String, String>() {

          public Iterable<String> call(String s) {
            String columns[] = s.split("\t");// "\\<.*\\>", " "
            String title = columns[1];
            String text = columns[3].replaceAll("\\<[^>]*>", " ")
                .replace("\\n", " ").toLowerCase()
                .replaceAll("[0-9]", " ")
                .replaceAll("[^a-zA-Z]", " ")
                .replaceAll("\\s+", " ");
            String words[] = text.split("\\s+");
            for (int i = 0; i < words.length; i++) {
              words[i] = words[i] + "," + title;
            }
            return Arrays.asList(words);
          }
        });

    // find the titles to count number of documents
    JavaRDD<String> titles = file
        .flatMap(new FlatMapFunction<String, String>() {
          public Iterable<String> call(String s) {
            return Arrays.asList(s.split("\t")[1]);
          }
        });
    final Long titleCount = titles.distinct().count();
    JavaPairRDD<String, Integer> pairs = wordTitle
        .mapToPair(new PairFunction<String, String, Integer>() {
          public Tuple2<String, Integer> call(String s) {
            return new Tuple2<String, Integer>(s, 1);
          }
        });
    // this is of the form word,document title, count
    JavaPairRDD<String, Integer> counts = pairs
        .reduceByKey(new Function2<Integer, Integer, Integer>() {
          public Integer call(Integer a, Integer b) {
            return a + b;
          }
        });
    // Count the number of documents in which each word occurs (d_word).
    // for the wordTitle each word will be counted once per document

    JavaPairRDD<String, Integer> d_word = counts
        .mapToPair(new PairFunction<Tuple2<String, Integer>, String, Integer>() {

          @Override
          public Tuple2<String, Integer> call(
              Tuple2<String, Integer> wordTitleCount)
              throws Exception {
            String wordTitle = wordTitleCount._1;
            String word = wordTitle.split(",")[0];

            return new Tuple2<String, Integer>(word, 1);
          }
        });
    JavaPairRDD<String, Integer> dwordCount = d_word
        .reduceByKey(new Function2<Integer, Integer, Integer>() {
          public Integer call(Integer a, Integer b) {
            return a + b;
          }
        });

    // Compute the IDF for each word-document pair, using log(N/d_word)
    // create another rdd having word as key and value as (document,count)
    JavaPairRDD<String, String> tempRdd = counts
        .mapToPair(new PairFunction<Tuple2<String, Integer>, String, String>() {

          @Override
          public Tuple2<String, String> call(
              Tuple2<String, Integer> wordTitleCount)
              throws Exception {
            String wordTitle = wordTitleCount._1;
            String word = wordTitle.split(",")[0];
            String title = wordTitle.split(",")[1];
            return new Tuple2<String, String>(word, title + ","
                + wordTitleCount._2);
          }
        });
    // join this temprdd and dword rdd to calculate idf and tf idf
    JavaPairRDD<String, Tuple2<String, Integer>> joined = tempRdd
        .join(dwordCount);

    // joined is of the form " word,title|count,dwordcount
    // for each word calculate idf, N is calculated above ,
    JavaPairRDD<String, String> idf = joined
        .mapToPair(new PairFunction<Tuple2<String, Tuple2<String, Integer>>, String, String>() {

          @Override
          public Tuple2<String, String> call(
              Tuple2<String, Tuple2<String, Integer>> mergedRecord)
              throws Exception {
            String word = mergedRecord._1;
            Tuple2<String, Integer> docCountDword = mergedRecord._2;
            String titleAndCount = docCountDword._1;
            String title = titleAndCount.split(",")[0];
            String tf = titleAndCount.split(",")[1];
            Integer dword = docCountDword._2;
            double IDF = Math.log(titleCount / dword);
            double tfidf = Integer.parseInt(tf) * IDF;
            return new Tuple2<String, String>(word, title + ","
                + tfidf);
          }

        });

    // find the records of word cloud

    List<String> cloudData = idf.lookup("cloud");
    // lookup returning an array which can be modified so create another
    // copy of that array
    List<String> temp = new ArrayList<String>(cloudData);
    cloudData = null;// free the memory
    // Runtime.getRuntime().gc();

    Collections.sort(temp, new Comparator<String>() {
      // each string is a combination of document name and tfidf
      @Override
      public int compare(String s1, String s2) {
        // get the tfidf and sort on that
        Double tfidf1 = Double.parseDouble(s1.split(",")[1]);
        Double tfidf2 = Double.parseDouble(s2.split(",")[1]);
        if (tfidf1.equals(tfidf2)) {

          return s1.split(",")[0].toLowerCase().compareTo(
              s2.split(",")[0].toLowerCase());
        }

        return tfidf2.compareTo(tfidf1);
      }
    });
    // write the arraylist to a file
    FileWriter fw = new FileWriter(new File("./output"));
    fw.write("total docs are " + titleCount + "\n");
    for (int i = 0; i < temp.size() && i < 100; i++) {
      String kv[] = temp.get(i).split(",");
      fw.write(kv[0] + "\t" + kv[1] + "\n");
    }
    fw.flush();
    fw.close();

  }
}
