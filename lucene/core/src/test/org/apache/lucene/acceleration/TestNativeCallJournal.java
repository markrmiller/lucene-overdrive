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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.PosixFilePermissions;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestNativeCallJournal extends LuceneTestCase {
  private Path privateDirectory() throws Exception {
    Path path = TrustedPathFixtures.directory("journal");
    Files.setPosixFilePermissions(path, PosixFilePermissions.fromString("rwx------"));
    return path;
  }

  public void testReplaceableAncestorCannotEraseQuarantineState() throws Exception {
    Path parent = privateDirectory();
    Path state = parent.resolve("state");
    Files.createDirectory(
        state, PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
    Files.setPosixFilePermissions(parent, PosixFilePermissions.fromString("rwxrwxrwx"));
    try {
      assertTrue(
          expectThrows(java.io.IOException.class, () -> new NativeCallJournal(state))
              .getMessage()
              .contains("ancestor"));
    } finally {
      Files.setPosixFilePermissions(parent, PosixFilePermissions.fromString("rwx------"));
    }
    try (var journal = new NativeCallJournal(state)) {
      assertFalse(journal.quarantined(NativeCallJournal.key("healthy")));
    }
  }

  public void testCapacityDenialDoesNotQuarantineOrPublish() throws Exception {
    Path directory = privateDirectory();
    String identity = "journal-capacity-admission";
    String key = NativeCallJournal.key(identity);
    var calls = new java.util.concurrent.atomic.AtomicInteger();
    var delegate =
        new AccelerationProvider.Kernel() {
          @Override
          public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
            return new AccelerationProvider.Estimate(true, 0, 0);
          }

          @Override
          public AccelerationProvider.Result execute(AccelerationProvider.Batch batch) {
            calls.incrementAndGet();
            batch.output().bytes().order(ByteOrder.LITTLE_ENDIAN).putLong(0, 99);
            return new AccelerationProvider.Result(AccelerationProvider.Status.OK, 1, 0);
          }
        };
    var output = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 1234);
    var batch =
        new AccelerationProvider.Batch(
            Capability.DOCSET_CARDINALITY,
            java.util.List.of(),
            new AccelerationProvider.Output(output, 1),
            ByteBuffer.allocate(64),
            ByteBuffer.allocate(0),
            ByteBuffer.allocate(256));
    try (var journal = new NativeCallJournal(directory)) {
      var wrapped = journal.wrap(delegate, identity);
      var slots = new java.util.ArrayList<Integer>();
      try {
        for (int i = 0; i < 256; i++) {
          slots.add(
              journal.begin(
                  NativeCallJournal.key("occupied-capacity-slot"),
                  Capability.DOCSET_CARDINALITY,
                  1));
        }
        assertTrue(
            expectThrows(
                    java.io.IOException.class,
                    () ->
                        journal.begin(
                            NativeCallJournal.key("full-capacity-probe"),
                            Capability.DOCSET_CARDINALITY,
                            1))
                .getMessage()
                .contains("capacity exhausted"));
        assertEquals(AccelerationProvider.Status.UNSUPPORTED, wrapped.execute(batch).status());
        assertEquals(0, calls.get());
        assertEquals(1234, output.getLong(0));
        assertFalse(journal.quarantined(key));
        journal.end(slots.remove(slots.size() - 1));
        assertEquals(AccelerationProvider.Status.OK, wrapped.execute(batch).status());
        assertEquals(1, calls.get());
        assertEquals(99, output.getLong(0));
        assertFalse(journal.quarantined(key));
      } finally {
        for (int slot : slots) journal.end(slot);
      }
    }
    try (var restarted = new NativeCallJournal(directory)) {
      assertFalse(restarted.quarantined(key));
    }
  }

  public void testDisappearedDiscoveredJournalIsNormalClose() throws Exception {
    Path directory = privateDirectory();
    Path candidate = directory.resolve("closing.journal");
    ByteBuffer data = ByteBuffer.allocate(16 + 256 * 88).order(ByteOrder.LITTLE_ENDIAN);
    data.putLong(0, Long.MAX_VALUE).putLong(8, 1);
    Files.write(candidate, data.array());
    NativeCallJournal.recover(
        directory,
        () -> {
          try {
            Files.delete(candidate);
          } catch (java.io.IOException error) {
            throw new java.io.UncheckedIOException(error);
          }
        });
    try (var journal = new NativeCallJournal(directory)) {
      assertFalse(journal.quarantined(NativeCallJournal.key("healthy")));
    }
  }

  public void testMalformedPublishedJournalStillRejected() throws Exception {
    Path directory = privateDirectory();
    Files.write(directory.resolve("broken.journal"), new byte[0]);
    expectThrows(java.io.IOException.class, () -> NativeCallJournal.recover(directory));
  }

  public void testPreparedJournalIsNotPublishedUntilHeaderReady() throws Exception {
    Path directory = privateDirectory();
    var prepared = new java.util.concurrent.CountDownLatch(1);
    var release = new java.util.concurrent.CountDownLatch(1);
    var task =
        new java.util.concurrent.FutureTask<>(
            () ->
                new NativeCallJournal(
                    directory,
                    () -> {
                      prepared.countDown();
                      try {
                        if (!release.await(20, java.util.concurrent.TimeUnit.SECONDS))
                          throw new AssertionError("publication was not released");
                      } catch (InterruptedException error) {
                        Thread.currentThread().interrupt();
                        throw new AssertionError(error);
                      }
                    }));
    Thread creator = new Thread(task, "journal-publication");
    creator.start();
    try {
      assertTrue(prepared.await(20, java.util.concurrent.TimeUnit.SECONDS));
      try (var paths = Files.list(directory)) {
        assertFalse(paths.anyMatch(path -> path.toString().endsWith(".journal")));
      }
      try (var concurrent = new NativeCallJournal(directory)) {
        NativeCallJournal.recover(directory);
        assertFalse(concurrent.quarantined(NativeCallJournal.key("preparing-owner")));
      }
    } finally {
      release.countDown();
    }
    try (var completed = task.get(20, java.util.concurrent.TimeUnit.SECONDS)) {
      NativeCallJournal.recover(directory);
      int slot =
          completed.begin(NativeCallJournal.key("published-owner"), Capability.OSQ_GATHER, 32);
      completed.end(slot);
    }
    creator.join(20000);
    assertFalse(creator.isAlive());
  }

  @org.apache.lucene.util.SuppressForbidden(reason = "ProcessBuilder log redirection requires File")
  public void testConcurrentProcessesOpenRecoverAndClose() throws Exception {
    Path directory = privateDirectory();
    String classpath =
        String.join(
            System.getProperty("path.separator"),
            Path.of(
                    NativeCallJournal.class
                        .getProtectionDomain()
                        .getCodeSource()
                        .getLocation()
                        .toURI())
                .toString(),
            Path.of(
                    NativeJournalRaceProbe.class
                        .getProtectionDomain()
                        .getCodeSource()
                        .getLocation()
                        .toURI())
                .toString());
    java.util.List<Process> processes = new java.util.ArrayList<>();
    java.util.List<Path> logs = new java.util.ArrayList<>();
    try {
      for (int i = 0; i < 3; i++) {
        Path log = directory.resolve("process-" + i + ".log");
        logs.add(log);
        processes.add(
            new ProcessBuilder(
                    Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-cp",
                    classpath,
                    NativeJournalRaceProbe.class.getName(),
                    directory.toString(),
                    Integer.toString(i),
                    "3")
                .redirectErrorStream(true)
                .redirectOutput(log.toFile())
                .start());
      }
      for (int i = 0; i < processes.size(); i++) {
        if (!processes.get(i).waitFor(60, java.util.concurrent.TimeUnit.SECONDS))
          fail("journal process timeout; outcome unknown");
        String log = Files.readString(logs.get(i));
        assertEquals(log, 0, processes.get(i).exitValue());
        assertTrue(log, log.contains("DONE_JOURNAL_PROCESS participant=" + i));
        System.out.println(log);
      }
    } finally {
      for (Process process : processes) if (process.isAlive()) process.destroyForcibly();
    }
    NativeCallJournal.recover(directory);
  }

  public void testActiveOwnerIsPreservedAndDeadCallIsQuarantined() throws Exception {
    Path directory = privateDirectory();
    String key = NativeCallJournal.key("provider:pack:capability:profile");
    try (var journal = new NativeCallJournal(directory)) {
      int slot = journal.begin(key, Capability.OSQ_GATHER, 32);
      NativeCallJournal.recover(directory);
      assertFalse(journal.quarantined(key));
      journal.end(slot);
    }
    ByteBuffer crashed = ByteBuffer.allocate(16 + 256 * 88).order(ByteOrder.LITTLE_ENDIAN);
    crashed.putLong(0, Long.MAX_VALUE).putLong(8, 1);
    crashed.putInt(16, 1).putInt(20, Capability.OSQ_GATHER.ordinal()).putLong(24, 32);
    crashed.put(32, key.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
    Files.write(directory.resolve("dead.journal"), crashed.array(), StandardOpenOption.CREATE_NEW);
    try (var recovered = new NativeCallJournal(directory)) {
      assertTrue(recovered.quarantined(key));
      assertTrue(
          Files.readString(directory.resolve(key + ".quarantine")).contains("outputElements=32\n"));
      assertFalse(recovered.quarantined(NativeCallJournal.key("different-pack")));
      assertFalse(Files.exists(directory.resolve("dead.journal")));
    }
    try (var restarted = new NativeCallJournal(directory)) {
      assertTrue(restarted.quarantined(key));
    }
  }

  public void testFailureQuarantineSurvivesRuntimeRestart() throws Exception {
    for (boolean declared : new boolean[] {false, true}) {
      Path directory = privateDirectory();
      String identity = "failed-provider-" + declared;
      String key = NativeCallJournal.key(identity);
      try (var journal = new NativeCallJournal(directory)) {
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        var kernel =
            journal.wrap(
                new AccelerationProvider.Kernel() {
                  @Override
                  public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
                    return new AccelerationProvider.Estimate(true, 0, 0);
                  }

                  @Override
                  public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
                      throws java.io.IOException {
                    calls.incrementAndGet();
                    if (!declared) throw new java.io.IOException("simulated provider fault");
                    return new AccelerationProvider.Result(
                        AccelerationProvider.Status.INTERNAL_ERROR, 0, 1);
                  }
                },
                identity);
        var batch =
            new AccelerationProvider.Batch(
                Capability.DOCSET_CARDINALITY,
                java.util.List.of(),
                new AccelerationProvider.Output(ByteBuffer.allocate(8), 1),
                ByteBuffer.allocate(64),
                ByteBuffer.allocate(0),
                ByteBuffer.allocate(256));
        if (declared)
          assertEquals(AccelerationProvider.Status.INTERNAL_ERROR, kernel.execute(batch).status());
        else expectThrows(java.io.IOException.class, () -> kernel.execute(batch));
        assertTrue(journal.quarantined(key));
        expectThrows(java.io.IOException.class, () -> kernel.execute(batch));
        assertEquals(1, calls.get());
      }
      try (var restarted = new NativeCallJournal(directory)) {
        assertTrue(restarted.quarantined(key));
      }
    }
  }

  public void testStartupCrashMetadataAndCapabilityIsolation() throws Exception {
    Path directory = privateDirectory();
    String digest = "a".repeat(64);
    String startup = AccelerationRuntime.BRIDGE_BUILD + ":" + digest + ":pack-startup";
    String startupKey = NativeCallJournal.key(startup);
    try (var journal = new NativeCallJournal(directory)) {
      int slot = journal.beginStartup(startup, digest);
      Path record;
      try (var paths = Files.list(directory)) {
        record =
            paths.filter(path -> path.toString().endsWith(".journal")).findFirst().orElseThrow();
      }
      try (var channel = java.nio.channels.FileChannel.open(record, StandardOpenOption.WRITE)) {
        channel.write(
            ByteBuffer.allocate(16)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(Long.MAX_VALUE)
                .putLong(1)
                .flip(),
            0);
      }
      NativeCallJournal.recover(directory);
      journal.end(slot);
      assertTrue(journal.quarantined(startupKey));
      String details = Files.readString(directory.resolve(startupKey + ".quarantine"));
      assertTrue(details.contains("capability=pack-startup"));
      assertTrue(details.contains("packDigest=" + digest));
      assertTrue(details.contains("bridge=" + AccelerationRuntime.BRIDGE_BUILD));
    }
    String first = digest + ":" + Capability.DOCSET_CARDINALITY.id + ":exact-integer-v1";
    String second = digest + ":" + Capability.DOCSET_INTERSECT_COUNT.id + ":exact-integer-v1";
    try (var journal = new NativeCallJournal(directory)) {
      journal.quarantine(NativeCallJournal.key(first));
      assertTrue(journal.quarantined(NativeCallJournal.key(first)));
      assertFalse(
          "integer profiles must retain independent capability identities",
          journal.quarantined(NativeCallJournal.key(second)));
    }
  }

  public void testCleanCompletionAndPrivateDirectory() throws Exception {
    Path directory = privateDirectory();
    String key = NativeCallJournal.key("successful-pack");
    try (var journal = new NativeCallJournal(directory)) {
      journal.end(journal.begin(key, Capability.DOCSET_INTERSECT_COUNT, 1024));
    }
    try (var restarted = new NativeCallJournal(directory)) {
      assertFalse(restarted.quarantined(key));
    }
    Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwxr-xr-x"));
    expectThrows(java.io.IOException.class, () -> new NativeCallJournal(directory));
  }
}
