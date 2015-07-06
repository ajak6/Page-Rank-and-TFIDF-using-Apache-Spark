import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class PageRank {
  public static void main(String[] args) {
    SparkConf sparkConf = new SparkConf().setAppName("Step 1 tfidf");
    JavaSparkContext spark = new JavaSparkContext(sparkConf);

    
    JavaRDD<String> file = spark
        .textFile("s3n://s15-p42-part2/wikipedia_arcs");
    
    JavaRDD<String> mappingFile = spark
        .textFile("s3n://s15-p42-part2/wikipedia_mapping");

    
    // merge the above records for same key to make adjacency list
    // eg:(6,[1852655, 999233, 1852663, 3167704, 3165770, 162624])
    JavaPairRDD<String, Iterable<String>> adjacencyList = file
        .mapToPair(new PairFunction<String, String, String>() {
          @Override
          public Tuple2<String, String> call(String s) {
            String[] parts = s.split("\t");
            return new Tuple2<String, String>(parts[0], parts[1]);
          }
        }).distinct().groupByKey().cache();
    JavaPairRDD<String, String> mappings = mappingFile.mapToPair(
        new PairFunction<String, String, String>() {
          @Override
          public Tuple2<String, String> call(String s) {
            String[] parts = s.split("\t");

            return new Tuple2<String, String>(parts[0], parts[1]);
          }
        }).distinct();
    /************************** dangling ****************************************/
    // find the dangling links from join of mapping and adjacency link
    // subtract will return the list of k v from this which does not have a
    // matching key in adjacency list
    JavaRDD<String> dangling = mappings.subtractByKey(adjacencyList).keys();
    // dangling.saveAsTextFile("hdfs:///join2");
    /****************************************************************************/
    /************ count total vertex for distributing rank of danglings *********/
    long totalVertices = mappingFile.distinct().count();
    /**************************************************************************/
    /************ assign initial score of 1 to all danglings ******************/

    JavaPairRDD<String, Double> danglingMap = dangling
        .mapToPair(new PairFunction<String, String, Double>() {

          @Override
          public Tuple2<String, Double> call(String s)
              throws Exception {
            return new Tuple2<String, Double>(s, 1.0);
          }

        });
    
    /******************************************************************************/
    JavaPairRDD<String, Double> ranks = adjacencyList
        .mapValues(new Function<Iterable<String>, Double>() {

          @Override
          public Double call(Iterable<String> rs) {
            return 1.0;
          }
        });
    
    // value: Return an RDD with the values of each tuple.
    // after adlist and ranks are joined RDD will be of the form :
    // (4,([3, 331725],1.0))
    // (6,([1852655, 999233, 1852663, 3167704, 3165770, 162624],1.0))
    
    // values function will give :([0, 4019482, 3084898, 1802545, 1802552],1.0)
    // 
    for (int current = 0; current < 10; current++) {
      // Calculates URL contributions to the rank of other URLs.
      JavaPairRDD<String, Double> contribs = adjacencyList
          .join(ranks)
          .values()
          .flatMapToPair(
              new PairFlatMapFunction<Tuple2<Iterable<String>, Double>, String, Double>() {
                @Override
                public Iterable<Tuple2<String, Double>> call(
                    Tuple2<Iterable<String>, Double> s) {
                  Iterable<String> size = s._1;
                  Iterator<String> it = size.iterator();
                  int urlCount = 0;
                  while (it.hasNext()) {
                    urlCount++;
                    it.next();
                  }

                  List<Tuple2<String, Double>> results = new ArrayList<Tuple2<String, Double>>();
                  for (String n : s._1) {
                    results.add(new Tuple2<String, Double>(
                        n, s._2() / urlCount));
                  }
                  return results;
                }
              });
      // calculate a common share for all nodes from dangling pointers
      // double sum = 0;
     
      final Accumulator<Double> accum = spark.accumulator(0.0);
      JavaRDD<Double> danlingRanks = danglingMap.values();
      // sum all valuse of danling ranks and add them to everyone
      // danlingRanks.saveAsTextFile("hdfs:///dv1"); // ---> this is
      // giving the rank of 3(dangling)
      // spark.parallelize(Arrays.asList(1, 2, 3, 4)).foreach(x ->
      // accum.add(x));
      // System.out.println("--------------size ----------" +
      // danlingRanks.count());
      Double sum = danlingRanks
          .reduce(new Function2<Double, Double, Double>() {

            @Override
            public Double call(Double x1, Double x2)
                throws Exception {
              return x1 + x2;
            }

          });
      accum.setValue(sum / totalVertices);
      
      
      final double addthis = accum.value();
      ranks = contribs.reduceByKey(new Sum()).mapValues(
          new Function<Double, Double>() {
            @Override
            public Double call(Double sum) {
              return 0.15 + (sum + addthis) * 0.85;
            }
          });
      JavaPairRDD<String, Tuple2<Double, Double>> dangOldNew = danglingMap
          .join(ranks);
      
      // update with new ranks of dangling vertex
      // if 7 is in dangling and is present in map and does not
      // have nayone pointing to it newRanks will return null
      // which breaks the logic: handle the null values or remove
      // such nodes from dangling list. first pass will tell you
      // which nodes does not have any incoming links also remove
      // them from dangling map and everything is win win.
      danglingMap = dangOldNew
          .mapToPair(new PairFunction<Tuple2<String, Tuple2<Double, Double>>, String, Double>() {
            public Tuple2<String, Double> call(
                Tuple2<String, Tuple2<Double, Double>> rec)
                throws Exception {
              String vertx = rec._1;
              Double newRank = rec._2()._2;
              return new Tuple2<String, Double>(vertx, newRank);
            }
          });

      // new dangling map
      
    }
    // sort rank rdd on values of double to get top 100
    // map(item => item.swap).sortByKey()
    JavaPairRDD<String, Tuple2<Double, String>> idRankName = ranks
        .join(mappings);
    JavaPairRDD<Double, Tuple2<String, String>> rankIdName = idRankName
        .mapToPair(
            new PairFunction<Tuple2<String, Tuple2<Double, String>>, Double, Tuple2<String, String>>() {

              @Override
              public Tuple2<Double, Tuple2<String, String>> call(
                  Tuple2<String, Tuple2<Double, String>> s)
                  throws Exception {
                String id = s._1;
                Tuple2<Double, String> rankName = s._2;
                Double rank = rankName._1;
                Tuple2<String, String> rightPart = new Tuple2<String, String>(
                    id, rankName._2);

                return new Tuple2<Double, Tuple2<String, String>>(
                    rank, rightPart);
              }

            }).sortByKey(false);
    
    List<Tuple2<Double, Tuple2<String, String>>> a = rankIdName.take(100);
    FileWriter fw;
    try {
      fw = new FileWriter(new File("./topPage"));
      for (int i = 0; i < a.size() && i < 100; i++) {
        
        String page = a.get(i)._2._2;
        Double rank = a.get(i)._1;
        fw.write(page + "\t" + rank + "\n");
      }
      fw.flush();
      fw.close();
    } catch (IOException e) {

      e.printStackTrace();
    }

  }

  private static class Sum implements Function2<Double, Double, Double> {
    @Override
    public Double call(Double a, Double b) {
      return a + b;
    }
  }
}
