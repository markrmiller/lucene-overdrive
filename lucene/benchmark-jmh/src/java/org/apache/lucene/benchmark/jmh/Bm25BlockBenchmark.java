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
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Bm25Block;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Prototype reader boundary for fused BM25 block scoring on a real mapped index. CURRENT_JAVA runs
 * the TermScorer block path: {@code NumericDocValues#longValues} over mapped norms, then the BM25
 * bulk scorer. NATIVE_OPTIMIZED replaces both with one kernel-29 call per block. FULL_TERM also
 * decodes the postings with {@code nextPostings}; NORMS_AND_SCORE replays predecoded blocks so only
 * the replaced work is timed. Each operation scores every block of one term.
 */
@SuppressForbidden(reason = "Benchmark temporary index and identity diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class Bm25BlockBenchmark {
  @Param({"CURRENT_JAVA", "NATIVE_OPTIMIZED"})
  public String implementation;

  /** "dense" occurs in about half of the documents, "sparse" in about 3%. */
  @Param({"dense", "sparse"})
  public String term;

  @Param({"FULL_TERM", "NORMS_AND_SCORE"})
  public String scope;

  private static final int DOCS = 100_000;

  private Path indexPath;
  private Directory directory;
  private DirectoryReader reader;
  private LeafReader leaf;
  private TermsEnum terms;
  private NumericDocValues norms;
  private Similarity.BulkSimScorer java;
  private AccelerationRuntime runtime;
  private Bm25Block block;
  private final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
  private long[] normValues = new long[0];
  private float[] scores = new float[0];
  private final List<int[]> blockDocs = new ArrayList<>();
  private final List<float[]> blockFreqs = new ArrayList<>();
  private float expected;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    indexPath = Files.createTempDirectory("lucene-bm25-block-jmh");
    try {
      directory = new MMapDirectory(indexPath);
      Random random = new Random(0x10529);
      try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
        StringBuilder text = new StringBuilder();
        for (int d = 0; d < DOCS; d++) {
          text.setLength(0);
          int length = 5 + random.nextInt(random.nextInt(10) == 0 ? 400 : 60);
          for (int w = 0; w < length; w++) text.append(" w").append(random.nextInt(2000));
          if (random.nextBoolean())
            for (int f = 1 + random.nextInt(3); f > 0; f--) text.append(" dense");
          if (random.nextInt(33) == 0) text.append(" sparse");
          Document doc = new Document();
          doc.add(new TextField("f", text.toString(), Field.Store.NO));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      reader = DirectoryReader.open(directory);
      leaf = reader.leaves().get(0).reader();
      terms = leaf.terms("f").iterator();
      if (!terms.seekExact(new Term("f", term).bytes())) throw new IllegalStateException(term);
      var searcher = new IndexSearcher(reader);
      var similarity = new BM25Similarity();
      var collection = searcher.collectionStatistics("f");
      var stats =
          searcher.termStatistics(new Term("f", term), terms.docFreq(), terms.totalTermFreq());
      java = similarity.scorer(1f, collection, stats).asBulkSimScorer();
      norms = leaf.getNormValues("f");
      PostingsEnum postings = terms.postings(null, PostingsEnum.FREQS);
      postings.nextDoc();
      while (postings.docID() != DocIdSetIterator.NO_MORE_DOCS) {
        postings.nextPostings(DocIdSetIterator.NO_MORE_DOCS, buffer);
        blockDocs.add(Arrays.copyOf(buffer.docs, buffer.size));
        blockFreqs.add(Arrays.copyOf(buffer.features, buffer.size));
      }
      expected = javaNormsAndScore();
      if (!implementation.equals("CURRENT_JAVA")) {
        runtime = BenchmarkAccelerationSupport.open(implementation, Capability.BM25_BLOCK);
        block = Bm25Block.prepare(runtime, leaf, "f", similarity, 1f, collection, stats);
        if (block == null) throw new IllegalStateException("norms are not mapped one-byte norms");
      }
      verifyParity();
    } catch (IOException | RuntimeException | Error failure) {
      closeResources();
      throw failure;
    }
  }

  @Benchmark
  public float score() throws IOException {
    if (scope.equals("NORMS_AND_SCORE"))
      return block == null ? javaNormsAndScore() : nativeNormsAndScore();
    PostingsEnum postings = terms.postings(null, PostingsEnum.FREQS);
    postings.nextDoc();
    float sum = 0;
    while (postings.docID() != DocIdSetIterator.NO_MORE_DOCS) {
      postings.nextPostings(DocIdSetIterator.NO_MORE_DOCS, buffer);
      sum += scoreBlock(buffer.size, buffer.docs, buffer.features);
    }
    return sum;
  }

  private float javaNormsAndScore() throws IOException {
    float sum = 0;
    for (int i = 0; i < blockDocs.size(); i++) {
      int[] docs = blockDocs.get(i);
      sum += javaBlock(docs.length, docs, blockFreqs.get(i));
    }
    return sum;
  }

  private float nativeNormsAndScore() throws IOException {
    float sum = 0;
    for (int i = 0; i < blockDocs.size(); i++) {
      int[] docs = blockDocs.get(i);
      sum += nativeBlock(docs.length, docs, blockFreqs.get(i));
    }
    return sum;
  }

  private float scoreBlock(int size, int[] docs, float[] freqs) throws IOException {
    return block == null ? javaBlock(size, docs, freqs) : nativeBlock(size, docs, freqs);
  }

  // Mirrors TermScorer#nextDocsAndScores after nextPostings; scores land in a separate array so
  // predecoded frequencies survive replay.
  private float javaBlock(int size, int[] docs, float[] freqs) throws IOException {
    if (normValues.length < size) {
      normValues = new long[ArrayUtil.oversize(size, Long.BYTES)];
      scores = new float[normValues.length];
    }
    norms.longValues(size, docs, normValues, 1L);
    java.score(size, freqs, normValues, scores);
    return scores[size - 1];
  }

  private float nativeBlock(int size, int[] docs, float[] freqs) throws IOException {
    if (scores.length < size) scores = new float[ArrayUtil.oversize(size, Float.BYTES)];
    block.score(size, docs, freqs, scores);
    return scores[size - 1];
  }

  private void verifyParity() throws IOException {
    if (block == null) return;
    for (int i = 0; i < blockDocs.size(); i++) {
      int[] docs = blockDocs.get(i);
      float[] freqs = blockFreqs.get(i);
      float[] want = new float[docs.length], got = new float[docs.length];
      long[] values = new long[docs.length];
      norms.longValues(docs.length, docs, values, 1L);
      java.score(docs.length, freqs, values, want);
      block.score(docs.length, docs, freqs, got);
      for (int j = 0; j < docs.length; j++)
        if (Float.floatToRawIntBits(want[j]) != Float.floatToRawIntBits(got[j]))
          throw new IllegalStateException("BM25 block parity mismatch at block " + i + " row " + j);
    }
    if (Float.floatToRawIntBits(nativeNormsAndScore()) != Float.floatToRawIntBits(expected))
      throw new IllegalStateException("BM25 block sum mismatch");
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    try {
      verifyParity();
      System.out.println("BM25 blocks per term: " + blockDocs.size());
    } finally {
      closeResources();
    }
    if (runtime != null) BenchmarkAccelerationSupport.verifyDrained(runtime);
  }

  private void closeResources() throws IOException {
    try {
      if (block != null) block.close();
    } finally {
      try {
        IOUtils.close(reader, directory);
      } finally {
        try {
          if (runtime != null) runtime.close();
        } finally {
          if (indexPath != null) IOUtils.rm(indexPath);
        }
      }
    }
  }
}
