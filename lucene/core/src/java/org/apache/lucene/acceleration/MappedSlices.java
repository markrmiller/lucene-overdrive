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
package org.apache.lucene.acceleration;

import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import org.apache.lucene.store.IndexInput;

/**
 * Direct views of mapped index bytes for kernels that read them in place. A view borrows the
 * mapping: callers must use it only while the owning reader is open.
 *
 * @lucene.experimental
 */
public final class MappedSlices {
  private MappedSlices() {}

  private static final class Access {
    static final MethodHandle SLICE = resolve();

    private static MethodHandle resolve() {
      try {
        return MethodHandles.publicLookup()
            .unreflect(
                Class.forName("org.apache.lucene.acceleration.MappedByteInput")
                    .getMethod("slice", IndexInput.class, long.class, long.class));
      } catch (ReflectiveOperationException failure) {
        throw new ExceptionInInitializerError(failure);
      }
    }
  }

  /**
   * Returns a direct view of {@code bytes} at {@code offset} of a memory-mapped input, or null when
   * the input is not mapped or the range spans mapping chunks.
   */
  public static ByteBuffer slice(IndexInput input, long offset, long bytes) throws IOException {
    try {
      return (ByteBuffer) Access.SLICE.invokeExact(input, offset, bytes);
    } catch (IOException | RuntimeException | Error failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IOException("mapped byte input unavailable", failure);
    }
  }
}
