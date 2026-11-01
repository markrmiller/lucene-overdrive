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
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.packed.PackedInts;
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
 * Benchmarks a cold {@link OrdinalMap#build} over sorted doc values, the per-term merge loop that
 * global ordinals and doc-values merges run.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(value = 2)
public class OrdinalMapBuildBenchmark {

  private static final String FIELD = "sdv";

  @Param({"1", "8"})
  public int segments;

  @Param({"200000"})
  public int termsPerSegment;

  private Path path;
  private Directory dir;
  private DirectoryReader reader;
  private SortedDocValues[] values;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    path = Files.createTempDirectory("ordinalMapBuildBench");
    dir = MMapDirectory.open(path);
    IndexWriterConfig iwc =
        new IndexWriterConfig()
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setRAMBufferSizeMB(1024);
    try (IndexWriter w = new IndexWriter(dir, iwc)) {
      // Half of each segment's terms are shared with the next segment.
      for (int s = 0; s < segments; s++) {
        int first = s * termsPerSegment / 2;
        for (int i = 0; i < termsPerSegment; i++) {
          Document doc = new Document();
          doc.add(
              new SortedDocValuesField(
                  FIELD, new BytesRef(String.format(Locale.ROOT, "%010d", first + i))));
          w.addDocument(doc);
        }
        w.flush();
      }
    }
    reader = DirectoryReader.open(dir);
    if (reader.leaves().size() != segments)
      throw new IllegalStateException("expected " + segments + " segments");
    values = new SortedDocValues[segments];
  }

  @Benchmark
  public long coldBuild() throws IOException {
    for (int i = 0; i < segments; i++)
      values[i] = reader.leaves().get(i).reader().getSortedDocValues(FIELD);
    try (OrdinalMap map = OrdinalMap.build(null, values, PackedInts.DEFAULT)) {
      return map.getValueCount();
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    reader.close();
    dir.close();
    IOUtils.rm(path);
  }
}
