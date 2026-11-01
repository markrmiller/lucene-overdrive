/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.codecs.lucene104;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.SuppressForbidden;

/** With no pack configured, indexing and searching never initialize the acceleration runtime. */
public class TestNoPackInitialization extends LuceneTestCase {

  @SuppressForbidden(reason = "Fork log redirection requires File")
  public void testIndexAndSearchLeaveRuntimeUninitialized() throws Exception {
    Path log = createTempFile("class-init", ".log");
    Path output = createTempFile("probe", ".out");
    List<String> command = new ArrayList<>();
    command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
    command.add("-Xlog:class+init=info:file=" + log);
    command.addAll(getJvmForkArguments());
    command.add(Probe.class.getName());
    Process process =
        new ProcessBuilder(command)
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    try {
      assertTrue("probe deadline", process.waitFor(60, TimeUnit.SECONDS));
      String out = Files.readString(output);
      assertEquals(out, 0, process.exitValue());
      assertTrue(out, out.contains("PROBE_OK"));
    } finally {
      if (process.isAlive()) process.destroyForcibly().waitFor(20, TimeUnit.SECONDS);
    }
    String initialized = Files.readString(log);
    // The probe reached the gated insert points.
    for (String exercised :
        new String[] {
          "org/apache/lucene/codecs/lucene104/Lucene104PostingsReader'",
          "org/apache/lucene/codecs/lucene104/Lucene104ScalarQuantizedVectorScorer'",
          "org/apache/lucene/util/quantization/OptimizedScalarQuantizer'"
        }) {
      assertTrue(exercised, initialized.contains("Initializing '" + exercised));
    }
    for (String untouched :
        new String[] {
          "org/apache/lucene/acceleration/AccelerationRuntime$Holder'",
          "org/apache/lucene/codecs/hnsw/AcceleratedFlatVectorsScorer'",
          "org/apache/lucene/internal/vectorization/AcceleratedVectorizationProvider'"
        }) {
      assertFalse(untouched, initialized.contains("Initializing '" + untouched));
    }
  }

  /** Runs in a fresh JVM with no acceleration properties. */
  public static final class Probe {
    public static void main(String[] args) throws Exception {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(
                  new Lucene104Codec() {
                    @Override
                    public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                      return new Lucene104HnswScalarQuantizedVectorsFormat();
                    }
                  });
      try (Directory directory = new ByteBuffersDirectory()) {
        try (IndexWriter writer = new IndexWriter(directory, config)) {
          for (int i = 0; i < 300; i++) {
            Document document = new Document();
            document.add(new TextField("body", "common term" + (i % 7), Field.Store.NO));
            document.add(
                new KnnFloatVectorField("vector", vector(i), VectorSimilarityFunction.EUCLIDEAN));
            writer.addDocument(document);
            if (i % 100 == 99) writer.flush();
          }
          writer.forceMerge(1);
        }
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          IndexSearcher searcher = new IndexSearcher(reader);
          if (searcher.search(new TermQuery(new Term("body", "common")), 10).totalHits.value()
                  != 300
              || searcher.search(new KnnFloatVectorQuery("vector", vector(3), 10), 10)
                      .scoreDocs
                      .length
                  != 10) throw new AssertionError("unexpected hits");
        }
      }
      System.out.println("PROBE_OK");
    }

    private static float[] vector(int i) {
      float[] vector = new float[32];
      for (int d = 0; d < vector.length; d++) vector[d] = (float) Math.sin(i * 31 + d);
      return vector;
    }
  }
}
