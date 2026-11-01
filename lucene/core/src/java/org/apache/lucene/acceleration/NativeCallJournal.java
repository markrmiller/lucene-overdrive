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
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicIntegerArray;
import org.apache.lucene.util.SuppressForbidden;

/**
 * Fixed private crash records: kernel identity, pack/ISA and output element count, never input
 * data.
 */
final class NativeCallJournal implements AutoCloseable {
  private static final int SLOTS = 256, HEADER = 16, STRIDE = 272;
  private static final int SIZE = HEADER + SLOTS * STRIDE;

  /** Admission failed before any native operation began; this is safe to replay in Java. */
  private static final class CapacityException extends IOException {
    private static final long serialVersionUID = 1L;

    CapacityException() {
      super("native crash journal capacity exhausted");
    }
  }

  private static final VarHandle STATE =
      MethodHandles.byteBufferViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);
  private final Path directory, path;
  private final MappedByteBuffer records;
  private final java.util.Map<String, String> descriptions =
      new java.util.concurrent.ConcurrentHashMap<>();
  private final AtomicIntegerArray busy = new AtomicIntegerArray(SLOTS);

  @SuppressForbidden(reason = "Private bounded native crash recovery uses process start identity")
  NativeCallJournal() throws IOException {
    this(
        Path.of(
            System.getProperty(
                "lucene.acceleration.native.statePath",
                Path.of(System.getProperty("user.home"), ".cache", "lucene-acceleration")
                    .toString())));
  }

  NativeCallJournal(Path directory) throws IOException {
    this(directory, null);
  }

  // Startup tests can pause before publication without exposing a partial journal to recovery.
  NativeCallJournal(Path directory, Runnable beforePublication) throws IOException {
    this.directory = directory;
    if (!directory.isAbsolute()) throw new IOException("native state path must be absolute");
    if (!Files.exists(directory, LinkOption.NOFOLLOW_LINKS))
      Files.createDirectories(
          directory,
          PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
    if (!Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS)
        || !Files.getPosixFilePermissions(directory)
            .equals(PosixFilePermissions.fromString("rwx------")))
      throw new IOException("native state directory must be private");
    TrustedPaths.verify(directory);
    recover(directory);
    path = directory.resolve(UUID.randomUUID() + ".journal");
    Path prepared =
        Files.createTempFile(
            directory,
            ".journal-",
            ".preparing",
            PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
    boolean published = false;
    try {
      try (FileChannel channel =
          FileChannel.open(prepared, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        records = channel.map(FileChannel.MapMode.READ_WRITE, 0, SIZE);
        records.order(ByteOrder.LITTLE_ENDIAN);
        records.putLong(0, ProcessHandle.current().pid());
        records.putLong(8, start(ProcessHandle.current()));
        records.force();
      }
      if (beforePublication != null) beforePublication.run();
      Files.move(prepared, path, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
      published = true;
    } finally {
      if (!published) Files.deleteIfExists(prepared);
    }
  }

  private static long start(ProcessHandle process) {
    return process.info().startInstant().orElse(Instant.EPOCH).toEpochMilli();
  }

  static String key(String identity) {
    try {
      return HexFormat.of()
          .formatHex(
              MessageDigest.getInstance("SHA-256")
                  .digest(identity.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError(impossible);
    }
  }

  static void recover(Path directory) throws IOException {
    recover(directory, null);
  }

  static void recover(Path directory, Runnable afterDiscovery) throws IOException {
    List<Path> paths;
    try (var stream = Files.list(directory)) {
      paths = stream.limit(1025).toList();
    }
    if (afterDiscovery != null) afterDiscovery.run();
    if (paths.size() > 1024) throw new IOException("native journal file bound exceeded");
    for (Path path : paths) {
      if (!path.getFileName().toString().endsWith(".journal")) continue;
      try {
        recoverFile(directory, path);
      } catch (
          @SuppressWarnings("unused")
          java.nio.file.NoSuchFileException disappeared) {
        // A live owner may close its journal after discovery, or another recovery may finish first.
      }
    }
  }

  private static void recoverFile(Path directory, Path path) throws IOException {
    var attributes =
        Files.readAttributes(
            path, java.nio.file.attribute.BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
    if (!attributes.isRegularFile()) throw new IOException("invalid native journal");
    ByteBuffer data;
    int stride;
    try (FileChannel channel =
        FileChannel.open(path, StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)) {
      long length = channel.size();
      if (length != SIZE && length != HEADER + SLOTS * 88)
        throw new IOException("invalid native journal");
      stride = length == SIZE ? STRIDE : 88;
      data = ByteBuffer.allocate(Math.toIntExact(length)).order(ByteOrder.LITTLE_ENDIAN);
      while (data.hasRemaining())
        if (channel.read(data) < 0) throw new IOException("truncated native journal");
    }
    var owner = ProcessHandle.of(data.getLong(0));
    if (owner.isPresent() && owner.get().isAlive() && start(owner.get()) == data.getLong(8)) return;
    for (int i = 0; i < SLOTS; i++) {
      int at = HEADER + i * stride;
      if (data.getInt(at) == 0) continue;
      byte[] key = new byte[64];
      data.get(at + 16, key);
      String identity = new String(key, java.nio.charset.StandardCharsets.US_ASCII);
      if (!identity.matches("[0-9a-f]{64}")) throw new IOException("invalid native crash identity");
      int ordinal = data.getInt(at + 4);
      if (ordinal < -1 || ordinal >= Capability.values().length)
        throw new IOException("invalid native crash capability");
      String capability = ordinal == -1 ? "pack-startup" : Capability.values()[ordinal].id;
      String details =
          "capability=" + capability + "\noutputElements=" + data.getLong(at + 8) + "\n";
      if (stride == STRIDE) {
        details +=
            "packDigest="
                + text(data, at + 80)
                + "\nbridge="
                + text(data, at + 144)
                + "\nisa="
                + text(data, at + 208)
                + "\n";
      } else details += "format=legacy\n";
      quarantine(directory, identity, details);
    }
    Files.deleteIfExists(path);
  }

  boolean quarantined(String identity) {
    return Files.exists(directory.resolve(identity + ".quarantine"), LinkOption.NOFOLLOW_LINKS);
  }

  void quarantine(String identity) throws IOException {
    quarantine(directory, identity, descriptions.getOrDefault(identity, "metadata=unavailable\n"));
  }

  private static void quarantine(Path directory, String identity, String details)
      throws IOException {
    Path target = directory.resolve(identity + ".quarantine");
    if (!Files.exists(target, LinkOption.NOFOLLOW_LINKS)) {
      try (FileChannel channel =
          FileChannel.open(target, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
        Files.setPosixFilePermissions(target, PosixFilePermissions.fromString("rw-------"));
        channel.write(
            ByteBuffer.wrap(
                ("native operation quarantined\n" + details)
                    .getBytes(java.nio.charset.StandardCharsets.US_ASCII)));
        channel.force(true);
      } catch (
          @SuppressWarnings("unused")
          java.nio.file.FileAlreadyExistsException ignored) {
        /* another owner persisted it */
      }
    }
  }

  int begin(String identity, Capability capability, long rows) throws IOException {
    return begin(metadata(identity, "0".repeat(64), "test"), capability, rows);
  }

  int beginStartup(String identity, String digest) throws IOException {
    String key = key(identity);
    byte[] metadata = metadata(key, digest, "startup");
    descriptions.put(
        key,
        "capability=pack-startup\npackDigest="
            + digest
            + "\nbridge="
            + AccelerationRuntime.BRIDGE_BUILD
            + "\n");
    return begin(metadata, null, 0);
  }

  private static byte[] metadata(String identity, String digest, String isa) {
    byte[] data = new byte[256];
    putText(data, 0, identity);
    putText(data, 64, digest);
    putText(data, 128, AccelerationRuntime.BRIDGE_BUILD);
    putText(data, 192, isa);
    return data;
  }

  private static void putText(byte[] destination, int offset, String text) {
    if (!text.matches("[a-zA-Z0-9._-]{1,64}"))
      throw new IllegalArgumentException("invalid native journal label");
    byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    System.arraycopy(bytes, 0, destination, offset, bytes.length);
  }

  private static String text(ByteBuffer data, int offset) throws IOException {
    byte[] bytes = new byte[64];
    data.get(offset, bytes);
    int length = 0;
    while (length < bytes.length && bytes[length] != 0) length++;
    String value = new String(bytes, 0, length, java.nio.charset.StandardCharsets.US_ASCII);
    if (!value.matches("[a-zA-Z0-9._-]{1,64}"))
      throw new IOException("invalid native journal metadata");
    return value;
  }

  private int begin(byte[] identity, Capability capability, long rows) throws IOException {
    for (int i = 0; i < SLOTS; i++) {
      if (!busy.compareAndSet(i, 0, 1)) continue;
      int at = HEADER + i * STRIDE;
      records.putInt(at + 4, capability == null ? -1 : capability.ordinal());
      records.putLong(at + 8, rows);
      records.put(at + 16, identity);
      STATE.setRelease(records, at, 1);
      return i;
    }
    throw new CapacityException();
  }

  void end(int slot) {
    STATE.setRelease(records, HEADER + slot * STRIDE, 0);
    busy.set(slot, 0);
  }

  AccelerationProvider.Kernel wrap(AccelerationProvider.Kernel kernel, String identity) {
    return wrap(kernel, identity, "0".repeat(64), "test", null);
  }

  AccelerationProvider.Kernel wrap(
      AccelerationProvider.Kernel kernel,
      String identity,
      String digest,
      String isa,
      Capability capability) {
    String key = key(identity);
    byte[] encodedKey = metadata(key, digest, isa);
    descriptions.put(
        key,
        "capability="
            + (capability == null ? "test" : capability.id)
            + "\npackDigest="
            + digest
            + "\nbridge="
            + AccelerationRuntime.BRIDGE_BUILD
            + "\nisa="
            + isa
            + "\n");
    var healthy = new java.util.concurrent.atomic.AtomicBoolean(true);
    return new AccelerationProvider.Kernel() {
      @Override
      public CriticalCallBudget.Snapshot criticalCalls() {
        return kernel.criticalCalls();
      }

      @Override
      public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
        return kernel.estimate(shape);
      }

      @Override
      public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
          throws IOException {
        if (!healthy.get()) throw new IOException("native kernel persistently quarantined");
        int slot;
        try {
          slot = begin(encodedKey, batch.capability(), batch.output().elements());
        } catch (
            @SuppressWarnings("unused")
            CapacityException denied) {
          return new AccelerationProvider.Result(AccelerationProvider.Status.UNSUPPORTED, 0, 0);
        }
        try {
          var result = kernel.execute(batch);
          if (result == null
              || result.status() == null
              || result.status() == AccelerationProvider.Status.INTERNAL_ERROR
              || result.status() == AccelerationProvider.Status.INVALID_INPUT) {
            healthy.set(false);
            quarantine(key);
          }
          return result;
        } catch (IOException | RuntimeException | Error error) {
          healthy.set(false);
          try {
            quarantine(key);
          } catch (IOException failure) {
            error.addSuppressed(failure);
          }
          throw error;
        } finally {
          end(slot);
        }
      }
    };
  }

  @Override
  public void close() throws IOException {
    for (int i = 0; i < SLOTS; i++)
      if (busy.get(i) != 0) throw new IOException("native journal still has active calls");
    records.force();
    Files.deleteIfExists(path);
  }
}
