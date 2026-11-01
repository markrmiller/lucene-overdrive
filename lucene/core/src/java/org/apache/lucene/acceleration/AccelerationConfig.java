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

import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Immutable startup policy. Installed packs use bounded capacity; zero disables allocation. */
public record AccelerationConfig(
    Mode mode,
    Path packPath,
    Set<Capability> required,
    Set<Capability> disabled,
    long nativeBytes,
    long requestBytes,
    int minimumBatch,
    boolean experimental,
    StoragePolicy storagePolicy,
    MemoryPolicy memoryPolicy,
    ShadowPolicy shadowPolicy,
    Tuning tuning) {
  public AccelerationConfig(
      Mode mode,
      Path packPath,
      Set<Capability> required,
      Set<Capability> disabled,
      long nativeBytes,
      long requestBytes,
      int minimumBatch,
      boolean experimental,
      StoragePolicy storagePolicy) {
    this(
        mode,
        packPath,
        required,
        disabled,
        nativeBytes,
        requestBytes,
        minimumBatch,
        experimental,
        storagePolicy,
        new MemoryPolicy(0, 0, 0, 0, 0),
        new ShadowPolicy(false, 1000, 0, 16),
        new Tuning(Map.of(), Map.of(), 0, 0));
  }

  /** Explicit enrolled capacity and advisory process headroom; zero disables enrollment. */
  public record MemoryPolicy(
      long heapBytes,
      long mappedBytes,
      long coreBurstBytes,
      long safetyHeadroomBytes,
      long processEnvelopeBytes,
      long ordinalBytes) {
    public MemoryPolicy(
        long heapBytes,
        long mappedBytes,
        long coreBurstBytes,
        long safetyHeadroomBytes,
        long processEnvelopeBytes) {
      this(heapBytes, mappedBytes, coreBurstBytes, safetyHeadroomBytes, processEnvelopeBytes, 0);
    }

    public MemoryPolicy {
      if (heapBytes < 0
          || mappedBytes < 0
          || coreBurstBytes < 0
          || safetyHeadroomBytes < 0
          || processEnvelopeBytes < 0
          || ordinalBytes < 0
          || ordinalBytes > heapBytes) throw new IllegalArgumentException("negative memory policy");
    }
  }

  /** Independent bounded secondary execution; its result is never published. */
  public record ShadowPolicy(
      boolean enabled, int sampleEvery, long maximumBytes, int mismatchLimit) {
    public ShadowPolicy {
      if (sampleEvery < 1
          || maximumBytes < 0
          || mismatchLimit < 0
          || mismatchLimit > 64
          || enabled && maximumBytes == 0)
        throw new IllegalArgumentException("invalid shadow policy");
    }
  }

  /** Qualified provider overrides and per-capability conservative cost/coverage tuning. */
  public record Tuning(
      Map<Capability, String> providerOverrides,
      Map<Capability, Integer> minimumBatches,
      long minimumBenefitNanos,
      long stagingNanosPerByte) {
    public Tuning {
      providerOverrides = Map.copyOf(providerOverrides);
      minimumBatches = Map.copyOf(minimumBatches);
      if (minimumBenefitNanos < 0
          || stagingNanosPerByte < 0
          || minimumBatches.values().stream().anyMatch(n -> n < 1)
          || providerOverrides.values().stream().anyMatch(String::isBlank))
        throw new IllegalArgumentException("invalid dispatch tuning");
    }
  }

  public long heapBytes() {
    return memoryPolicy.heapBytes();
  }

  public long mappedBytes() {
    return memoryPolicy.mappedBytes();
  }

  /** Compute selection. */
  public enum Mode {
    OFF,
    AUTO,
    REQUIRED
  }

  /** Storage selection is independent of compute. */
  public enum StoragePolicy {
    CURRENT,
    SELECTIVE_OFF_HEAP
  }

  public AccelerationConfig {
    required = Set.copyOf(required);
    disabled = Set.copyOf(disabled);
    if (mode == null
        || storagePolicy == null
        || memoryPolicy == null
        || shadowPolicy == null
        || tuning == null
        || nativeBytes < 0
        || requestBytes < 0
        || minimumBatch < 1) throw new IllegalArgumentException("invalid acceleration policy");
    if ((mode == Mode.REQUIRED && required.isEmpty())
        || (mode != Mode.REQUIRED && !required.isEmpty()))
      throw new IllegalArgumentException("required mode needs an explicit nonempty capability set");
    if (required.stream().anyMatch(disabled::contains))
      throw new IllegalArgumentException("a capability is both required and disabled");
    if (required.contains(Capability.TEST_COPY))
      throw new IllegalArgumentException("test capability cannot be a production requirement");
    if (packPath != null && !packPath.isAbsolute())
      throw new IllegalArgumentException("pack path must be absolute");
  }

  /** Default no-pack behavior, resolved exactly once by the process registry. */
  public static AccelerationConfig fromSystemProperties() {
    String prefix = "lucene.acceleration.";
    String path = System.getProperty(prefix + "native.packPath", "");
    String storage = System.getProperty(prefix + "storage.policy", "current");
    long defaultNative = Math.min(64L << 20, Runtime.getRuntime().maxMemory() / 128);
    long nativeBytes = longProperty(prefix + "memory.nativeBytes", defaultNative);
    return new AccelerationConfig(
        switch (System.getProperty(prefix + "native.mode", "auto")) {
          case "off" -> Mode.OFF;
          case "auto" -> Mode.AUTO;
          case "required" -> Mode.REQUIRED;
          default -> throw new IllegalArgumentException("unknown native mode");
        },
        path.isEmpty() ? null : Path.of(path),
        parseCapabilities(System.getProperty(prefix + "native.requiredCapabilities", "")),
        parseCapabilities(System.getProperty(prefix + "native.disabledCapabilities", "")),
        nativeBytes,
        longProperty(prefix + "memory.requestBytes", Math.min(8L << 20, nativeBytes / 4)),
        Integer.parseInt(System.getProperty(prefix + "native.minimumBatch", "1")),
        booleanProperty(prefix + "experimental.enabled", true),
        switch (storage) {
          case "current" -> StoragePolicy.CURRENT;
          case "selectiveOffHeap" -> StoragePolicy.SELECTIVE_OFF_HEAP;
          default -> throw new IllegalArgumentException("unknown storage policy");
        },
        new MemoryPolicy(
            longProperty(prefix + "memory.managedHeapBytes", 0),
            longProperty(prefix + "memory.mappedBytes", 0),
            longProperty(prefix + "memory.coreBurstBytes", 0),
            longProperty(prefix + "memory.safetyHeadroomBytes", 0),
            longProperty(prefix + "memory.processEnvelopeBytes", 0),
            longProperty(prefix + "memory.ordinalBytes", 0)),
        new ShadowPolicy(
            booleanProperty(prefix + "shadow.enabled", false),
            Integer.parseInt(System.getProperty(prefix + "shadow.sampleEvery", "1000")),
            longProperty(prefix + "shadow.maximumBytes", 0),
            Integer.parseInt(System.getProperty(prefix + "shadow.mismatchLimit", "16"))),
        parseTuning(prefix));
  }

  private static boolean booleanProperty(String key, boolean fallback) {
    return switch (System.getProperty(key, Boolean.toString(fallback))) {
      case "true" -> true;
      case "false" -> false;
      default -> throw new IllegalArgumentException("invalid boolean policy: " + key);
    };
  }

  private static long longProperty(String key, long fallback) {
    return Long.parseLong(System.getProperty(key, Long.toString(fallback)));
  }

  private static Tuning parseTuning(String prefix) {
    Map<Capability, String> overrides = new HashMap<>();
    Map<Capability, Integer> batches = new HashMap<>();
    for (Capability capability : Capability.values()) {
      String override = System.getProperty(prefix + "native.provider." + capability.id);
      if (override != null) overrides.put(capability, override);
      String batch = System.getProperty(prefix + "native.minimumBatch." + capability.id);
      if (batch != null) batches.put(capability, Integer.parseInt(batch));
    }
    return new Tuning(
        overrides,
        batches,
        longProperty(prefix + "native.minimumBenefitNanos", 0),
        longProperty(prefix + "native.stagingNanosPerByte", 0));
  }

  private static Set<Capability> parseCapabilities(String value) {
    Set<Capability> result = new HashSet<>();
    if (!value.isBlank()) {
      for (String id : value.split(",", -1)) {
        if (!result.add(Capability.fromId(id.trim())))
          throw new IllegalArgumentException("duplicate capability");
      }
    }
    return result;
  }
}
