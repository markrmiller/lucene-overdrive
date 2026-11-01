/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
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

import org.apache.lucene.index.VectorSimilarityFunction;

/** Structural bounds of the reusable mapped byte scorer; qualification lives in data. */
final class MeasuredByteProfile {
  private MeasuredByteProfile() {}

  static final int MAXIMUM_ROWS = 512;

  static boolean shape(int dimensions, int rows, VectorSimilarityFunction similarity) {
    return dimensions > 0
        && dimensions <= 32768
        && rows > 1
        && rows <= MAXIMUM_ROWS
        && similarity == VectorSimilarityFunction.DOT_PRODUCT;
  }
}
