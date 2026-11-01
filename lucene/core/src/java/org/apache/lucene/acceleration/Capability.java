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
package org.apache.lucene.acceleration;

/** Fixed, versioned operation registry. All native implementations are experimental. */
public enum Capability {
  TEST_COPY("test.copy.u64.v1", 1, "exact-integer-v1"),
  VECTOR_I8("vector.score.i8.exact.v1", 2, "lucene-10.5.2-i8-v1"),
  DOCSET_INTERSECT("docset.intersect.words.exact.v1", 3, "exact-integer-v1"),
  DOCSET_UNION("docset.union.words.exact.v1", 4, "exact-integer-v1"),
  DOCSET_DIFFERENCE("docset.difference.words.exact.v1", 5, "exact-integer-v1"),
  DOCSET_CARDINALITY("docset.cardinality.words.exact.v1", 6, "exact-integer-v1"),
  DOCVALUES_DECODE("docvalues.decode.exact.v1", 7, "lucene-10.5.2-direct-le-v1"),
  DOCVALUES_RANGE("docvalues.range.exact.v1", 8, "exact-integer-v1"),
  POSTINGS_SPLIT("postings.decode.exact.v1", 9, "lucene-10.5.2-split-v1"),
  QUANTIZE_PACK("vector.quantize.pack.exact.v1", 10, "lucene-10.5.2-nibble-v1"),
  FACETS_COUNT("facets.count.exact.v1", 11, "exact-integer-v1"),
  DOCVALUES_AGGREGATE("docvalues.decode.filter.aggregate.exact.v1", 12, "exact-integer-v1"),
  QUANTIZED_RESCORE("vector.rescore.quantized.exact.v1", 13, "lucene-10.5.2-u7-v1"),
  QUANTIZE_TRANSPOSE("vector.quantize.transpose.exact.v1", 14, "lucene-10.5.2-bitplanes-v1"),
  SORTED_INTERSECT("docset.intersect.sorted.exact.v1", 15, "exact-integer-v1"),
  MULTI_INTERSECT("docset.intersect.multi.exact.v1", 16, "exact-integer-v1"),
  SPARSE_FACETS("facets.count.sparse.exact.v1", 17, "exact-integer-v1"),
  MODERN_QUANTIZED_RESCORE("vector.rescore.lvq.exact.v1", 18, "lucene-10.5.2-lvq-v1"),
  OPTIMIZED_QUANTIZE("vector.quantize.lvq.exact.v1", 19, "lucene-10.5.2-lvq-quantize-v1"),
  VECTOR_STATISTICS("vector.statistics.exact.v1", 20, "lucene-10.5.2-ordered-f32-v1"),
  POSTINGS_PREFIX("postings.prefixsum.exact.v1", 21, "exact-integer-v1"),
  QUANTIZE_UNPACK("vector.quantize.unpack.exact.v1", 22, "lucene-10.5.2-nibble-v1"),
  QUANTIZE_DIBIT("vector.quantize.dibit.exact.v1", 23, "lucene-10.5.2-bitplanes-v1"),
  QUANTIZE_BINARY("vector.quantize.binary.exact.v1", 24, "lucene-10.5.2-bitplanes-v1"),
  POSTINGS_BLOCK("postings.block.decode.exact.v1", 25, "lucene-10.5.2-forutil-v1"),
  FLOATING_STATS("docvalues.stats.f64.exact.v1", 26, "lucene-10.5.2-welford-f64-v1"),
  DOCSET_INTERSECT_COUNT("docset.intersect.count.words.exact.v1", 27, "exact-integer-v1"),
  DOCVALUES_RANGE_FILTER("docvalues.range.filter.words.exact.v1", 28, "exact-integer-v1"),
  BM25_BLOCK("similarity.bm25.block.exact.v1", 29, "lucene-10.5.2-bm25-v1"),
  OSQ_GATHER("vector.rescore.lvq.gather.exact.v1", 30, "lucene-10.5.2-lvq-v1"),
  MASKED_FACETS("facets.count.masked.exact.v1", 31, "exact-integer-v1"),
  VECTOR_F32("vector.score.f32.exact.v1", 100, "current-java-provider-v1"),
  MAXSIM("maxsim.f32.exact.v1", 101, "solr-maxsim-v1"),
  COMPRESSION("compression.decode.exact.v1", 102, "lucene-10.5.2-lz4-v1");

  public final String id;
  public final int kernelId;
  public final String arithmeticProfile;

  Capability(String id, int kernelId, String arithmeticProfile) {
    this.id = id;
    this.kernelId = kernelId;
    this.arithmeticProfile = arithmeticProfile;
  }

  /** Rejects unknown configuration keys rather than silently weakening requirements. */
  public static Capability fromId(String id) {
    for (Capability capability : values()) {
      if (capability.id.equals(id)) return capability;
    }
    throw new IllegalArgumentException("unknown acceleration capability: " + id);
  }
}
