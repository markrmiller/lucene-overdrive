/* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.internal.vectorization;

import java.io.IOException;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.codecs.hnsw.AcceleratedFlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.store.IndexInput;

/** Source-owned composition; retains the guarded provider's original Java implementations. */
final class AcceleratedVectorizationProvider extends VectorizationProvider {
  private final VectorizationProvider java;

  AcceleratedVectorizationProvider(VectorizationProvider java) {
    this.java = java;
  }

  @Override
  public VectorUtilSupport getVectorUtilSupport() {
    return java.getVectorUtilSupport();
  }

  @Override
  public String description() {
    return java.description();
  }

  private FlatVectorsScorer scorer(FlatVectorsScorer delegate) {
    AccelerationRuntime runtime = AccelerationRuntime.system();
    return runtime.supports(Capability.VECTOR_I8)
            || runtime.supports(Capability.QUANTIZED_RESCORE)
            || runtime.supports(Capability.VECTOR_F32)
        ? new AcceleratedFlatVectorsScorer(delegate, runtime)
        : delegate;
  }

  @Override
  public FlatVectorsScorer getLucene99FlatVectorsScorer() {
    return scorer(java.getLucene99FlatVectorsScorer());
  }

  @Override
  public FlatVectorsScorer getLucene99ScalarQuantizedVectorsScorer() {
    return scorer(java.getLucene99ScalarQuantizedVectorsScorer());
  }

  @Override
  public PostingDecodingUtil newPostingDecodingUtil(IndexInput input) throws IOException {
    return java.newPostingDecodingUtil(input);
  }

  @Override
  public DocValuesBulkDecodeSupport getDocValuesBulkDecodeSupport() {
    return java.getDocValuesBulkDecodeSupport();
  }

  @Override
  public DocValuesRangeSupport getDocValuesRangeSupport() {
    return java.getDocValuesRangeSupport();
  }
}
