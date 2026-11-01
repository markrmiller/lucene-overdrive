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

import java.io.IOException;
import java.nio.file.Path;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.NativeAllocator;

/** Internal implementation boundary keeping FFM classes out of the Java-only startup path. */
public interface NativeBridge extends NativeAllocator {
  /** Allocator for staging that is allocated, used and closed on the calling thread. */
  default NativeAllocator synchronousAllocator() {
    return this;
  }

  /** Classifies actual scoped storage before production qualification admits a batch. */
  default boolean isMapped(AccelerationProvider.Batch batch) {
    return false;
  }

  /** Validates platform minima before loading any pack code. Empty minima impose no extra bound. */
  default void verifyPlatform(String minimumOs, String minimumLibc) throws IOException {
    if (!minimumOs.isEmpty()
        && compareVersion(System.getProperty("os.version", "0"), minimumOs) < 0)
      throw new IOException("kernel version below pack minimum");
    if (!minimumLibc.isEmpty())
      throw new IOException("libc minimum cannot be verified by this bridge");
  }

  /** Validates a compatible v1 negotiation prefix; unknown feature extensions remain explicit. */
  static long negotiatedFeatures(
      int status,
      int size,
      int major,
      int minor,
      int target,
      long features,
      long reserved,
      int expectedTarget)
      throws IOException {
    if (status != 0
        || size != 32
        || major != 1
        || minor < 0
        || target != expectedTarget
        || reserved != 0) throw new IOException("native ABI/target mismatch");
    if ((features & ~3L) != 0) throw new IOException("UNSUPPORTED_FEATURE_EXTENSION");
    return features;
  }

  /** Short-call policy: expensive exact float reductions never block safepoints. */
  static boolean criticalCallAllowed(Capability capability, long workBytes) {
    return workBytes >= 0
        && workBytes <= 256 * 1024
        && capability != Capability.VECTOR_F32
        && capability != Capability.MAXSIM;
  }

  /** Numeric version comparison ignores a distribution suffix after its numeric prefix. */
  static int compareVersion(String actual, String minimum) {
    if (!minimum.matches("[0-9]{1,9}(\\.[0-9]{1,9}){0,7}"))
      throw new IllegalArgumentException("invalid platform minimum");
    String actualPrefix = actual.split("[^0-9.]", 2)[0];
    if (!actualPrefix.matches("[0-9]{1,9}(\\.[0-9]{1,9}){0,7}"))
      throw new IllegalArgumentException("unrecognized host platform version");
    String[] a = actualPrefix.split("\\."), b = minimum.split("\\.");
    for (int i = 0; i < Math.max(a.length, b.length); i++) {
      long left = i < a.length ? Long.parseLong(a[i]) : 0;
      long right = i < b.length ? Long.parseLong(b[i]) : 0;
      if (left != right) return Long.compare(left, right);
    }
    return 0;
  }

  /** Loads a verified absolute library in a process-lifetime arena and negotiates ABI v1. */
  AccelerationProvider.Prepared prepare(
      Path library, AccelerationProvider.Descriptor descriptor, MemoryLedger ledger)
      throws IOException;
}
