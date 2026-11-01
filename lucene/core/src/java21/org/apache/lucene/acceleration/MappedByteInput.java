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

import java.io.IOException;
import java.nio.ByteBuffer;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;

/**
 * Mapped input access for explicit byte-batch qualification; the caller must pin its reader.
 *
 * @lucene.experimental
 */
public final class MappedByteInput {
  private MappedByteInput() {}

  /** Returns a borrowed contiguous native view, or null when this mapping cannot provide one. */
  public static ByteBuffer slice(IndexInput input, long offset, long length) throws IOException {
    input = FilterIndexInput.unwrapOnlyTest(input);
    if (!(input instanceof MemorySegmentAccessInput mapped)) return null;
    var segment = mapped.segmentSliceOrNull(offset, length);
    return segment == null ? null : segment.asByteBuffer();
  }
}
