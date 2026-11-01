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
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.ArrayList;
import java.util.List;

/** Explicit trusted deployment data; no CPU model names and no unmeasured capability enrollment. */
final class QualificationPolicy {
  record Profile(
      String digest,
      String isa,
      Capability capability,
      int dimensions,
      long minimumRows,
      long maximumRows,
      Long parameter3,
      Long parameter4,
      String layout,
      long maximumBytes,
      String evidence,
      String consumerContext) {
    boolean shape(AccelerationProvider.Shape shape, boolean mapped) {
      return dimensions == shape.dimensions()
          && shape.elements() >= minimumRows
          && shape.elements() <= maximumRows
          && shape.bytes() <= maximumBytes
          && (layout.equals("mapped")
              ? mapped && shape.nativeAddressable()
              : layout.equals("native") ? !mapped && shape.nativeAddressable() : !mapped);
    }

    boolean context(String consumer) {
      return consumerContext.equals(consumer);
    }

    boolean batch(AccelerationProvider.Batch batch) {
      var parameters = batch.parameters().duplicate().order(ByteOrder.LITTLE_ENDIAN);
      return parameters(parameters.getLong(24), parameters.getLong(32));
    }

    boolean parameters(long supplied3, long supplied4) {
      return (parameter3 == null || parameter3 == supplied3)
          && (parameter4 == null || parameter4 == supplied4);
    }
  }

  private static final List<String> RANGE_CONTEXTS =
      List.of("generic", "solr.point.uncached.full-live.v1", "solr.frange.full-live.v1");

  private static final List<String> BYTE_CONTEXTS =
      List.of(
          "generic",
          OwnedByteBatchScorer.EXACT_OWNED_CONTEXT,
          OwnedByteBatchScorer.EXACT_STANDALONE_CONTEXT,
          OwnedByteBatchScorer.EXACT_STANDALONE_PACKED_CONTEXT);

  private static final List<String> AGGREGATE_CONTEXTS =
      List.of("generic", "solr.json-range.full-live.v1");

  final List<Profile> profiles;

  private QualificationPolicy(List<Profile> profiles) {
    this.profiles = List.copyOf(profiles);
  }

  static QualificationPolicy empty() {
    return new QualificationPolicy(List.of());
  }

  QualificationPolicy append(QualificationPolicy other) throws IOException {
    if (profiles.size() + other.profiles.size() > 256)
      throw new IOException("too many bundled qualification profiles");
    List<Profile> combined = new ArrayList<>(profiles);
    combined.addAll(other.profiles);
    return new QualificationPolicy(combined);
  }

  static QualificationPolicy load() throws IOException {
    String path = System.getProperty("lucene.acceleration.native.qualificationPath", "");
    return path.isEmpty() ? empty() : read(Path.of(path));
  }

  static QualificationPolicy read(Path path) throws IOException {
    if (!path.isAbsolute()
        || !Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)
        || Files.size(path) > 65536) throw new IOException("invalid qualification file");
    try {
      var permissions = Files.getPosixFilePermissions(path, LinkOption.NOFOLLOW_LINKS);
      if (permissions.contains(PosixFilePermission.GROUP_WRITE)
          || permissions.contains(PosixFilePermission.OTHERS_WRITE))
        throw new IOException("writable qualification file");
    } catch (UnsupportedOperationException ignored) {
      throw new IOException("qualification permissions cannot be verified", ignored);
    }
    for (Path part = path; part != null; part = part.getParent()) {
      if (Files.isSymbolicLink(part)) throw new IOException("qualification symlink");
    }
    TrustedPaths.verify(path);
    byte[] bytes;
    try (var stream = Files.newInputStream(path)) {
      bytes = stream.readNBytes(65537);
    }
    if (bytes.length > 65536) throw new IOException("qualification bounds");
    return parse(
        java.nio.charset.StandardCharsets.UTF_8
            .newDecoder()
            .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
            .decode(java.nio.ByteBuffer.wrap(bytes))
            .toString());
  }

  // TSV: bridge, digest, ISA, capability, dimensions, min/max rows, parameter3/4,
  // layout (mapped/native/buffer), max bytes, evidence label, optional consumer context. '-' means
  // an unused parameter.
  static QualificationPolicy parse(String text) throws IOException {
    if (text.length() > 65536) throw new IOException("qualification bounds");
    List<Profile> result = new ArrayList<>();
    try {
      for (String line : text.split("\\R")) {
        if (line.isBlank() || line.startsWith("#")) continue;
        String[] fields = line.split("\\t", -1);
        if (fields.length != 12 && fields.length != 13
            || result.size() >= 256
            || !fields[0].equals(AccelerationRuntime.BRIDGE_BUILD)
            || !fields[1].matches("[0-9a-f]{64}")
            || !fields[2].matches("[a-z0-9-]{1,64}")
            || !fields[11].matches("[a-zA-Z0-9._/-]{1,160}")
            || fields.length == 13 && !validConsumerContext(fields[12]))
          throw new IOException("invalid qualification row");
        var capability = Capability.fromId(fields[3]);
        String consumer = fields.length == 13 ? fields[12] : "generic";
        if (fields.length == 13
            && !(capability == Capability.DOCVALUES_RANGE_FILTER
                    && RANGE_CONTEXTS.contains(consumer)
                || capability == Capability.VECTOR_I8 && BYTE_CONTEXTS.contains(consumer)
                || capability == Capability.DOCVALUES_AGGREGATE
                    && AGGREGATE_CONTEXTS.contains(consumer)))
          throw new IOException("invalid consumer context");
        if ((capability == Capability.VECTOR_I8 || capability == Capability.DOCVALUES_AGGREGATE)
            && !consumer.equals("generic")
            && !fields[9].equals("mapped"))
          throw new IOException("scoped mapped consumer requires mapped inputs");
        int dimensions = Integer.parseInt(fields[4]);
        long minimum = Long.parseLong(fields[5]), maximum = Long.parseLong(fields[6]);
        long bytes = Long.parseLong(fields[10]);
        if (capability == Capability.TEST_COPY
            || dimensions < 0
            || minimum < 1
            || maximum < minimum
            || maximum > 16L * 1024 * 1024
            || bytes < 0
            || bytes > Integer.MAX_VALUE
            || !List.of("mapped", "native", "buffer").contains(fields[9]))
          throw new IOException("invalid qualification shape");
        result.add(
            new Profile(
                fields[1],
                fields[2],
                capability,
                dimensions,
                minimum,
                maximum,
                fields[7].equals("-") ? null : Long.valueOf(fields[7]),
                fields[8].equals("-") ? null : Long.valueOf(fields[8]),
                fields[9],
                bytes,
                fields[11],
                consumer));
      }
    } catch (IllegalArgumentException error) {
      throw new IOException("invalid qualification row", error);
    }
    return new QualificationPolicy(result);
  }

  static boolean validConsumerContext(String consumer) {
    if (consumer == null || consumer.isEmpty() || consumer.length() > 64) return false;
    for (int i = 0; i < consumer.length(); i++) {
      char c = consumer.charAt(i);
      if (!(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '.' || c == '_' || c == '-'))
        return false;
    }
    return true;
  }

  List<Profile> forKernel(String digest, String isa, Capability capability) {
    return profiles.stream()
        .filter(
            profile ->
                profile.digest().equals(digest)
                    && profile.isa().equals(isa)
                    && profile.capability() == capability)
        .toList();
  }
}
