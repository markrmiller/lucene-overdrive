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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Startup verification, immutable selection and staged native failure contracts. */
public class TestAccelerationRuntime extends LuceneTestCase {
  static AccelerationConfig config(
      AccelerationConfig.Mode mode,
      Path pack,
      Set<Capability> required,
      Set<Capability> disabled,
      int minimumBatch) {
    return new AccelerationConfig(
        mode,
        pack,
        required,
        disabled,
        16L << 20,
        8L << 20,
        minimumBatch,
        true,
        AccelerationConfig.StoragePolicy.CURRENT);
  }

  static Path nativePack() {
    String path = System.getProperty("tests.acceleration.pack");
    assumeTrue("optional native reference pack was not supplied", path != null);
    assumeTrue("native FFM tests require JDK25", Runtime.version().feature() >= 25);
    return Path.of(path).toAbsolutePath();
  }

  public void testCompatibleNegotiatedMinorAndExplicitFeatureExtension() throws Exception {
    assertEquals(3L, NativeBridge.negotiatedFeatures(0, 32, 1, 1, 1, 3, 0, 1));
    expectThrows(IOException.class, () -> NativeBridge.negotiatedFeatures(0, 32, 2, 0, 1, 3, 0, 1));
    expectThrows(
        IOException.class, () -> NativeBridge.negotiatedFeatures(0, 32, 1, -1, 1, 3, 0, 1));
    assertTrue(
        expectThrows(
                IOException.class, () -> NativeBridge.negotiatedFeatures(0, 32, 1, 1, 1, 4, 0, 1))
            .getMessage()
            .contains("UNSUPPORTED_FEATURE_EXTENSION"));
  }

  public void testCriticalCallPolicyRetiresExpensiveFloatReductions() {
    assertFalse(NativeBridge.criticalCallAllowed(Capability.VECTOR_F32, 1));
    assertFalse(NativeBridge.criticalCallAllowed(Capability.MAXSIM, 1));
    assertTrue(NativeBridge.criticalCallAllowed(Capability.OSQ_GATHER, 32768));
    assertFalse(NativeBridge.criticalCallAllowed(Capability.OSQ_GATHER, 256 * 1024 + 1));
    assertFalse(NativeBridge.criticalCallAllowed(Capability.DOCSET_CARDINALITY, -1));
  }

  public void testSlowCriticalCallRetiresOnlyItsBoundKernel() {
    var slow = new CriticalCallBudget();
    var other = new CriticalCallBudget();
    assertTrue(slow.allows(Capability.DOCSET_CARDINALITY, 32));
    slow.record(CriticalCallBudget.MAXIMUM_NANOS);
    assertTrue(slow.allows(Capability.DOCSET_CARDINALITY, 32));
    slow.record(CriticalCallBudget.MAXIMUM_NANOS + 1);
    assertFalse(slow.allows(Capability.DOCSET_CARDINALITY, 32));
    assertTrue(other.allows(Capability.DOCSET_CARDINALITY, 32));
    slow.record(1);
    assertFalse(slow.allows(Capability.DOCSET_CARDINALITY, 32));
    assertEquals(3, slow.snapshot().calls());
    assertEquals(1, slow.snapshot().slowCalls());
    assertEquals(CriticalCallBudget.MAXIMUM_NANOS + 1, slow.snapshot().maximumNanos());
    assertTrue(slow.snapshot().disabled());
    expectThrows(IllegalArgumentException.class, () -> slow.record(-1));
  }

  public void testPeerWritableAncestorCannotReplaceVerifiedPackOrPolicy() throws Exception {
    Path pack = fakePack(false);
    Path parent = TrustedPathFixtures.directory("replaceable-pack-parent");
    Path child = parent.resolve("pack");
    Files.move(pack, child);
    Path policy = parent.resolve("qualification.tsv");
    Files.writeString(policy, "# no production enrollment\n");
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    Files.setPosixFilePermissions(parent, PosixFilePermissions.fromString("rwxrwxrwx"));
    try {
      assertTrue(
          expectThrows(IOException.class, () -> PackLoader.verify(child))
              .getMessage()
              .contains("ancestor"));
      assertTrue(
          expectThrows(IOException.class, () -> QualificationPolicy.read(policy))
              .getMessage()
              .contains("ancestor"));
    } finally {
      Files.setPosixFilePermissions(parent, PosixFilePermissions.fromString("rwx------"));
    }
    assertFalse(PackLoader.verify(child).capabilities().isEmpty());
    assertTrue(QualificationPolicy.read(policy).profiles.isEmpty());
  }

  public void testTrustRejectsSymlinkAncestorsAndIgnoresUsernameProperties() throws Exception {
    Path pack = fakePack(false);
    Path parent = TrustedPathFixtures.directory("pack-alias");
    Path alias = parent.resolve("alias");
    Files.createSymbolicLink(alias, pack);
    expectThrows(IOException.class, () -> PackLoader.verify(alias));
    Path policy = pack.resolve("qualification.tsv");
    Files.writeString(policy, "# no production enrollment\n");
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    assertTrue(
        expectThrows(
                IOException.class,
                () -> QualificationPolicy.read(alias.resolve("qualification.tsv")))
            .getMessage()
            .contains("symlink"));
    String previous = System.getProperty("user.name");
    try {
      System.setProperty("user.name", "not-the-effective-uid");
      assertNotNull(PackLoader.verify(pack));
      assertTrue(QualificationPolicy.read(policy).profiles.isEmpty());
    } finally {
      if (previous == null) System.clearProperty("user.name");
      else System.setProperty("user.name", previous);
    }
  }

  public void testJavaOnlyAndOffDoNotInspectPack() throws Exception {
    var absent =
        new AccelerationRuntime(config(AccelerationConfig.Mode.AUTO, null, Set.of(), Set.of(), 1));
    assertEquals(AccelerationRuntime.State.JAVA_ONLY, absent.inventory().state());
    assertFalse(absent.hasKernels());
    assertTrue(absent.inventory().ready());
    assertFalse(absent.eligible(Capability.VECTOR_I8, 100));
    assertEquals(
        1L,
        absent
            .inventory()
            .counters()
            .get(Capability.VECTOR_I8)
            .fallbacks()
            .get(AccelerationRuntime.Reason.NO_PACK)
            .longValue());
    Path malformed = createTempDir("off-pack");
    Files.writeString(malformed.resolve("manifest.json"), "this is not JSON");
    var off =
        new AccelerationRuntime(
            config(AccelerationConfig.Mode.OFF, malformed, Set.of(), Set.of(), 1));
    assertEquals(AccelerationRuntime.State.JAVA_ONLY, off.inventory().state());
    assertTrue(off.inventory().rejections().isEmpty());
    assertFalse(off.eligible(Capability.VECTOR_I8, 100));
    assertEquals(
        1L,
        off.inventory()
            .counters()
            .get(Capability.VECTOR_I8)
            .fallbacks()
            .get(AccelerationRuntime.Reason.OFF)
            .longValue());
    assertEquals(Long.valueOf(0), off.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testRequiredAndDisabledConfiguration() throws Exception {
    expectThrows(
        IllegalArgumentException.class,
        () -> config(AccelerationConfig.Mode.REQUIRED, null, Set.of(), Set.of(), 1));
    expectThrows(
        IllegalArgumentException.class,
        () ->
            config(
                AccelerationConfig.Mode.REQUIRED,
                null,
                Set.of(Capability.VECTOR_I8),
                Set.of(Capability.VECTOR_I8),
                1));
    expectThrows(IllegalArgumentException.class, () -> Capability.fromId("unknown.operation.v1"));
    expectThrows(
        IOException.class,
        () ->
            new AccelerationRuntime(
                config(
                    AccelerationConfig.Mode.REQUIRED,
                    null,
                    Set.of(Capability.VECTOR_I8),
                    Set.of(),
                    1)));
  }

  public void testSynchronousStagingIsConfinedAndOrdinaryStagingRemainsShared() throws Exception {
    try (var runtime =
            AccelerationRuntime.forQualification(
                config(AccelerationConfig.Mode.AUTO, nativePack(), Set.of(), Set.of(), 1));
        var confined = runtime.newCallScope();
        var shared = runtime.newScope()) {
      ByteBuffer callBuffer = confined.allocate(16);
      ByteBuffer sharedBuffer = shared.allocate(16);
      Thread other =
          new Thread(
              () -> {
                RuntimeException failure =
                    expectThrows(RuntimeException.class, () -> callBuffer.putInt(0, 1));
                assertEquals("java.lang.WrongThreadException", failure.getClass().getName());
                expectThrows(IllegalStateException.class, confined::close);
                runtime.retryFailedScratchReclamations();
                sharedBuffer.putInt(0, 42);
              });
      other.start();
      other.join(10_000);
      assertFalse(other.isAlive());
      callBuffer.putInt(0, 7);
      assertEquals(7, callBuffer.getInt(0));
      assertEquals(42, sharedBuffer.getInt(0));
    }
  }

  @SuppressWarnings("try") // the task binding scopes frame creation
  public void testCallFramesAreReusedPerThreadAndRetired() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            config(AccelerationConfig.Mode.AUTO, nativePack(), Set.of(), Set.of(), 1));
    CallFrame frame;
    try {
      var task =
          new AccelerationRuntime.TaskContext(new MemoryLedger.Owner("test", "core", 0, "task"), 0);
      try (var binding = runtime.enterTask(task)) {
        frame = runtime.acquireFrame();
        assertNotNull(frame);
        assertNull("frames are not reentrant", runtime.acquireFrame());
        frame.data().putInt(CallFrame.DATA_BYTES - 4, 9);
        assertEquals(64, frame.parameters().remaining());
        frame.close();
      }
      task.cancel();
      assertFalse("a task never owns the frame", frame.scope().isCancelled());
      try (CallFrame again = runtime.acquireFrame()) {
        assertSame(frame, again);
        assertEquals(9, again.data().getInt(CallFrame.DATA_BYTES - 4));
      }

      CallFrame[] other = new CallFrame[1];
      Thread worker =
          new Thread(
              () -> {
                try (CallFrame own = runtime.acquireFrame()) {
                  other[0] = own;
                  expectThrows(IllegalStateException.class, frame::close);
                } catch (IOException e) {
                  throw new AssertionError(e);
                }
              });
      worker.start();
      worker.join(10_000);
      assertFalse(worker.isAlive());
      assertNotNull(other[0]);
      assertNotSame(frame, other[0]);
      assertFalse(other[0].isRetired());
      Thread successor =
          new Thread(
              () -> {
                try (CallFrame own = runtime.acquireFrame()) {
                  assertNotNull(own);
                } catch (IOException e) {
                  throw new AssertionError(e);
                }
              });
      successor.start();
      successor.join(10_000);
      assertFalse(successor.isAlive());
      assertTrue("an exited thread's frame is freed by the next new frame", other[0].isRetired());

      // A busy frame holds shutdown until its owner closes it, which frees it.
      CallFrame busy = runtime.acquireFrame();
      assertSame(frame, busy);
      Throwable[] closeFailure = new Throwable[1];
      Thread closer =
          new Thread(
              () -> {
                try {
                  runtime.close();
                } catch (Throwable t) {
                  closeFailure[0] = t;
                }
              });
      closer.start();
      long deadline = System.nanoTime() + 10_000_000_000L;
      while (!runtime.isStopping() && System.nanoTime() < deadline) Thread.onSpinWait();
      assertTrue(runtime.isStopping());
      assertFalse(busy.isRetired());
      busy.close();
      closer.join(10_000);
      assertFalse(closer.isAlive());
      assertNull(closeFailure[0]);
      assertTrue(busy.isRetired()); // shutdown returned, so the ledger also drained
      expectThrows(IOException.class, runtime::acquireFrame);
    } finally {
      runtime.close();
    }
  }

  public void testStrictManifestParser() throws Exception {
    for (String value :
        List.of(
            "{\"a\":1,\"a\":2}",
            "{} trailing",
            "{\"a\":1.5}",
            "[1,]",
            "\"\\u+066\"",
            "\"\\u-066\"",
            "{\"a\":\u0661}",
            "{\"x\":\"bad\nstring\"}")) {
      expectThrows(IOException.class, () -> BoundedJson.parse(value));
    }
    expectThrows(IOException.class, () -> BoundedJson.parse(" ".repeat(65537)));
    expectThrows(IOException.class, () -> BoundedJson.parse("[".repeat(18) + "0" + "]".repeat(18)));
  }

  public void testPassiveVerificationRejectsUnsafeArtifacts() throws Exception {
    assumeTrue("reference manifest targets Linux", System.getProperty("os.name").equals("Linux"));
    Path root = fakePack(false);
    String manifest = Files.readString(root.resolve("manifest.json"));
    assertTrue(PackLoader.verify(root).capabilities().contains(Capability.VECTOR_I8));
    Files.writeString(root.resolve("manifest.json"), manifest.replace("native.so", "../native.so"));
    assertTrue(
        expectThrows(IOException.class, () -> PackLoader.verify(root))
            .getMessage()
            .contains("unsafe"));
    Files.writeString(root.resolve("manifest.json"), manifest);
    Files.write(root.resolve("native.so"), new byte[] {9, 8});
    assertTrue(
        expectThrows(IOException.class, () -> PackLoader.verify(root))
            .getMessage()
            .contains("checksum"));
    Path injected = fakePack(true);
    assertTrue(
        expectThrows(IOException.class, () -> PackLoader.verify(injected))
            .getMessage()
            .contains("API"));
  }

  public void testPassiveVerificationRejectsOverflowPriorityAndNestedContractFields()
      throws Exception {
    assumeTrue("reference manifest targets Linux", System.getProperty("os.name").equals("Linux"));
    for (String invalidPriority : List.of("2147483648", "-2147483649", "9223372036854775807")) {
      Path root = fakePack(false);
      Path manifest = root.resolve("manifest.json");
      Files.writeString(
          manifest,
          Files.readString(manifest)
              .replace("\"priority\":100", "\"priority\":" + invalidPriority));
      assertTrue(
          expectThrows(IOException.class, () -> PackLoader.verify(root))
              .getMessage()
              .contains("priority"));
      var runtime =
          AccelerationRuntime.forQualification(
              config(AccelerationConfig.Mode.AUTO, root, Set.of(), Set.of(), 1));
      assertEquals(AccelerationRuntime.State.JAVA_ONLY, runtime.inventory().state());
      assertTrue(
          runtime.inventory().rejections().toString(),
          runtime.inventory().rejections().stream().anyMatch(r -> r.startsWith("VERIFY:")));
    }
    for (String field : List.of("api", "nativeAbi")) {
      Path root = fakePack(false);
      Path manifest = root.resolve("manifest.json");
      Files.writeString(
          manifest,
          Files.readString(manifest)
              .replace(
                  "\"" + field + "\":{\"major\":1,\"minor\":0}",
                  "\"" + field + "\":{\"major\":1,\"minor\":0,\"unverifiedFeature\":true}"));
      assertTrue(
          expectThrows(IOException.class, () -> PackLoader.verify(root))
              .getMessage()
              .contains("unknown"));
    }
  }

  public void testAdditiveMinorMetadataAndOptionalKernels() throws Exception {
    Path root = fakePack(false);
    Path manifest = root.resolve("manifest.json");
    String original = Files.readString(manifest);
    String additive =
        original
            .replace("\"minor\":0", "\"minor\":1")
            .replace(
                "\"manifestSchema\":1",
                "\"manifestSchema\":1,\"buildMetadata\":{\"producer\":\"future\"}")
            .replace(
                "\"kernels\":[",
                "\"kernels\":[{\"operation\":\"future.optional.v1\",\"optional\":true},");
    Files.writeString(manifest, additive);
    assertEquals(Set.of(Capability.VECTOR_I8), PackLoader.verify(root).capabilities());
    Files.writeString(manifest, additive.replace("\"optional\":true", "\"optional\":false"));
    expectThrows(IOException.class, () -> PackLoader.verify(root));
    Files.writeString(
        manifest,
        additive.replace(
            "\"buildMetadata\":", "\"requiredFields\":[\"buildMetadata\"],\"buildMetadata\":"));
    expectThrows(IOException.class, () -> PackLoader.verify(root));
    Files.writeString(manifest, additive.replace("\"major\":1", "\"major\":2"));
    expectThrows(IOException.class, () -> PackLoader.verify(root));
  }

  public void testTemporaryParentTrustAllowsRootStickyButRejectsPeers() throws Exception {
    TrustedPathFixtures.assumeHostAccess();
    TrustedPaths.verifyTemporaryParent(Path.of("/tmp"));
    Path directory = TrustedPathFixtures.directory("temporary-parent");
    TrustedPaths.verifyTemporaryParent(directory);
    Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwxrwxrwx"));
    try {
      expectThrows(IOException.class, () -> TrustedPaths.verifyTemporaryParent(directory));
    } finally {
      Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwx------"));
    }
    Path alias = directory.resolve("alias");
    Files.createSymbolicLink(alias, Path.of("/tmp"));
    expectThrows(IOException.class, () -> TrustedPaths.verifyTemporaryParent(alias));
  }

  @org.apache.lucene.util.SuppressForbidden(reason = "Fork log redirection requires File")
  public void testStartupSnapshotUsesTrustedFallbackFromUnsafeTmpdir() throws Exception {
    Path pack = fakePack(false);
    Path parent = TrustedPathFixtures.directory("unsafe-startup-tmp-parent");
    Path unsafe = parent.resolve("tmp");
    Files.createDirectory(
        unsafe, PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
    Path output = Path.of(createTempFile("snapshot-startup", ".log").toString());
    String classpath =
        String.join(
            System.getProperty("path.separator"),
            Path.of(
                    AccelerationRuntime.class
                        .getProtectionDomain()
                        .getCodeSource()
                        .getLocation()
                        .toURI())
                .toString(),
            Path.of(
                    SnapshotStartupProbe.class
                        .getProtectionDomain()
                        .getCodeSource()
                        .getLocation()
                        .toURI())
                .toString());
    Files.setPosixFilePermissions(parent, PosixFilePermissions.fromString("rwxrwxrwx"));
    try {
      var process =
          new ProcessBuilder(
                  Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                  "-Djava.io.tmpdir=" + unsafe,
                  "-cp",
                  classpath,
                  SnapshotStartupProbe.class.getName(),
                  pack.toString())
              .redirectErrorStream(true)
              .redirectOutput(output.toFile())
              .start();
      try {
        assertTrue(
            "snapshot probe deadline", process.waitFor(20, java.util.concurrent.TimeUnit.SECONDS));
        String log = Files.readString(output);
        assertEquals(log, 0, process.exitValue());
        assertTrue(log, log.contains("SNAPSHOT_OK:" + PackLoader.verify(pack).digest()));
        try (var entries = Files.list(unsafe)) {
          assertEquals(0, entries.count());
        }
      } finally {
        if (process.isAlive()) {
          process.destroyForcibly();
          assertTrue(process.waitFor(20, java.util.concurrent.TimeUnit.SECONDS));
        }
      }
    } finally {
      Files.setPosixFilePermissions(parent, PosixFilePermissions.fromString("rwx------"));
    }
  }

  public void testVerifiedSnapshotSurvivesSourceReplacementAndRejectsChangedBytes()
      throws Exception {
    Path root = fakePack(false);
    var verified = PackLoader.verify(root);
    var snapshot = PackLoader.snapshot(verified);
    try {
      Path source = root.resolve("native.so");
      byte[] frozen = Files.readAllBytes(snapshot.library());
      Files.writeString(source, "replacement");
      assertArrayEquals(frozen, Files.readAllBytes(snapshot.library()));
      assertEquals(verified.digest(), PackLoader.verify(snapshot.root()).digest());
      expectThrows(IOException.class, () -> PackLoader.snapshot(verified));
      Files.writeString(root.resolve("manifest.json"), "different manifest");
      expectThrows(IOException.class, () -> PackLoader.snapshot(verified));
    } finally {
      PackLoader.deleteSnapshot(snapshot.root());
    }
  }

  public void testPassiveVerificationRejectsMalformedUtf8() throws Exception {
    assumeTrue("reference manifest targets Linux", System.getProperty("os.name").equals("Linux"));
    Path root = fakePack(false);
    Path manifest = root.resolve("manifest.json");
    byte[] bytes = Files.readAllBytes(manifest);
    int value =
        Files.readString(manifest).indexOf("\"packId\":\"fixture\"") + "\"packId\":\"".length();
    assertTrue(value >= "\"packId\":\"".length());
    bytes[value] = (byte) 0xc3;
    bytes[value + 1] = 0x28;
    Files.write(manifest, bytes);
    expectThrows(java.nio.charset.CharacterCodingException.class, () -> PackLoader.verify(root));
  }

  public void testPassiveVerificationRejectsUnverifiedJarClassPath() throws Exception {
    assumeTrue("reference manifest targets Linux", System.getProperty("os.name").equals("Linux"));
    Path root = fakePack(false);
    Path jarPath = root.resolve("provider.jar");
    String oldDigest = digest(jarPath);
    var jarManifest = new java.util.jar.Manifest();
    jarManifest.getMainAttributes().putValue("Manifest-Version", "1.0");
    jarManifest.getMainAttributes().putValue("Class-Path", "unverified.jar");
    try (var jar = new JarOutputStream(Files.newOutputStream(jarPath), jarManifest)) {
      jar.putNextEntry(new JarEntry("marker.txt"));
      jar.write(1);
      jar.closeEntry();
    }
    Path manifest = root.resolve("manifest.json");
    Files.writeString(manifest, Files.readString(manifest).replace(oldDigest, digest(jarPath)));
    assertTrue(
        expectThrows(IOException.class, () -> PackLoader.verify(root))
            .getMessage()
            .contains("class path"));
  }

  public void testPassiveVerificationRejectsPeerWritableArtifactsAndSubdirectories()
      throws Exception {
    assumeTrue("reference manifest targets Linux", System.getProperty("os.name").equals("Linux"));
    Path root = fakePack(false);
    assumeTrue(
        "requires POSIX permissions", Files.getFileStore(root).supportsFileAttributeView("posix"));
    Path library = root.resolve("native.so");
    Files.setPosixFilePermissions(library, PosixFilePermissions.fromString("rw-rw-r--"));
    assertTrue(
        expectThrows(IOException.class, () -> PackLoader.verify(root))
            .getMessage()
            .contains("writable by peers"));
    Files.setPosixFilePermissions(library, PosixFilePermissions.fromString("rw-r--r--"));
    Path nested = root.resolve("nested");
    Files.createDirectory(nested);
    Files.move(library, nested.resolve("native.so"));
    Files.setPosixFilePermissions(nested, PosixFilePermissions.fromString("rwxrwxr-x"));
    Path manifest = root.resolve("manifest.json");
    Files.writeString(
        manifest,
        Files.readString(manifest)
            .replace("\"path\":\"native.so\"", "\"path\":\"nested/native.so\""));
    assertTrue(
        expectThrows(IOException.class, () -> PackLoader.verify(root))
            .getMessage()
            .contains("writable by peers"));
    Files.setPosixFilePermissions(nested, PosixFilePermissions.fromString("rwxr-xr-x"));
    assertNotNull(PackLoader.verify(root));
  }

  public void testCompatiblePackAndFrozenInventory() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            config(
                AccelerationConfig.Mode.AUTO,
                nativePack(),
                Set.of(),
                Set.of(Capability.FACETS_COUNT),
                8));
    assertEquals(
        runtime.inventory().rejections().toString(),
        AccelerationRuntime.State.FROZEN,
        runtime.inventory().state());
    assertTrue(runtime.supports(Capability.VECTOR_I8));
    assertFalse(runtime.supports(Capability.FACETS_COUNT));
    // A packaged kernel that fails its own fixtures is dropped silently; the reference pack must
    // pass every one.
    for (String rejection : runtime.inventory().rejections())
      assertFalse(rejection, rejection.startsWith("SELF_TEST:"));
    assertTrue(
        runtime.inventory().rejections().toString(), runtime.supports(Capability.MASKED_FACETS));
    assertTrue(runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "generic"));
    assertTrue(
        runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "solr.point.uncached.full-live.v1"));
    assertTrue(runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "solr.frange.full-live.v1"));
    assertFalse(runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "bad context"));
    assertTrue(
        runtime.inventory().rejections().toString(), runtime.supports(Capability.VECTOR_F32));
    assertTrue(runtime.inventory().rejections().toString(), runtime.supports(Capability.MAXSIM));
    expectThrows(UnsupportedOperationException.class, () -> runtime.inventory().selected().clear());
    assertFalse(runtime.eligible(Capability.VECTOR_I8, 7));
    assertTrue(runtime.eligible(Capability.VECTOR_I8, 8));
    assertFalse(runtime.eligible(Capability.FACETS_COUNT, 100));
    assertEquals(
        1L,
        runtime
            .inventory()
            .counters()
            .get(Capability.VECTOR_I8)
            .fallbacks()
            .get(AccelerationRuntime.Reason.SMALL_BATCH)
            .longValue());
    var required =
        AccelerationRuntime.forQualification(
            config(
                AccelerationConfig.Mode.REQUIRED,
                nativePack(),
                Set.of(Capability.VECTOR_I8),
                Set.of(),
                1));
    assertTrue(required.inventory().ready());
    try (var floatRequired =
        AccelerationRuntime.forQualification(
            config(
                AccelerationConfig.Mode.REQUIRED,
                nativePack(),
                Set.of(Capability.VECTOR_F32),
                Set.of(),
                1))) {
      assertTrue(floatRequired.supports(Capability.VECTOR_F32));
    }
  }

  /** Heap batches require the live short-call route; retirement and large batches decline. */
  public void testHeapBuffersUseDirectCriticalCall() throws Exception {
    try (var runtime =
            AccelerationRuntime.forQualification(
                config(AccelerationConfig.Mode.AUTO, nativePack(), Set.of(), Set.of(), 1));
        var scope = runtime.newScope()) {
      assumeTrue("pack lacks intersect", runtime.supports(Capability.DOCSET_INTERSECT));
      long declines = 0;
      for (int words : new int[] {1, 7, 64, 4096, 16384}) {
        byte[] left = new byte[words * 8 + 3],
            right = new byte[words * 8],
            out = new byte[words * 8];
        random().nextBytes(left);
        random().nextBytes(right);
        // A sliced heap input checks that the array offset reaches the kernel.
        ByteBuffer a = ByteBuffer.wrap(left, 3, words * 8).slice().order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer b = ByteBuffer.wrap(right).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer o = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        parameters.putLong(0, words).putLong(16, words * 64L);
        var batch =
            new AccelerationProvider.Batch(
                Capability.DOCSET_INTERSECT,
                List.of(
                    new AccelerationProvider.Input(a, words, AccelerationProvider.Encoding.I64),
                    new AccelerationProvider.Input(b, words, AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(o, words),
                parameters,
                scope.allocate(0),
                scope.allocate(256));
        long calls = intersectCounter(runtime).nativeCalls();
        boolean admitted =
            3L * words * 8 <= 256 * 1024
                && !runtime.criticalCalls().get(Capability.DOCSET_INTERSECT).disabled();
        assertEquals(admitted, runtime.execute(batch, words * 64L, 0));
        assertEquals(calls + (admitted ? 1 : 0), intersectCounter(runtime).nativeCalls());
        if (admitted) {
          for (int i = 0; i < words; i++)
            assertEquals(a.getLong(i * 8) & b.getLong(i * 8), o.getLong(i * 8));
        } else {
          declines++;
          for (byte v : out) assertEquals(0, v);
        }
      }
      assertEquals(
          declines,
          intersectCounter(runtime)
              .fallbacks()
              .get(AccelerationRuntime.Reason.UNSUPPORTED)
              .longValue());
      assertTrue(runtime.supports(Capability.DOCSET_INTERSECT));
    }
  }

  private static AccelerationRuntime.Counter intersectCounter(AccelerationRuntime runtime) {
    return runtime.inventory().counters().get(Capability.DOCSET_INTERSECT);
  }

  public void testAbiMismatchRejectedBeforeSelection() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            config(AccelerationConfig.Mode.AUTO, faultPack(1), Set.of(), Set.of(), 1));
    assertFalse(runtime.hasKernels());
    assertFalse(runtime.inventory().rejections().isEmpty());
  }

  public void testEqualPriorityProvidersDeclineAmbiguousCapabilities() throws Exception {
    Path canonical = nativePack();
    Path collection = localPackDirectory("ambiguous-packs");
    copyPack(canonical, collection.resolve("first"));
    copyPack(canonical, collection.resolve("second"));
    var runtime =
        AccelerationRuntime.forQualification(
            config(AccelerationConfig.Mode.AUTO, collection, Set.of(), Set.of(), 1));
    assertFalse(runtime.hasKernels());
    assertTrue(
        runtime.inventory().rejections().toString(),
        runtime.inventory().rejections().contains("AMBIGUOUS:" + Capability.VECTOR_I8.id));
    assertEquals(AccelerationRuntime.State.JAVA_ONLY, runtime.inventory().state());
  }

  public void testNativeErrorsQuarantineAndRequiredReadiness() throws Exception {
    checkFault(2, false);
    checkFault(2, true);
    checkFault(3, false);
    checkFault(4, false);
  }

  private void checkFault(int fault, boolean required) throws Exception {
    String previousState = System.getProperty("lucene.acceleration.native.statePath");
    Path state = TrustedPathFixtures.directory("fault-state");
    Files.setPosixFilePermissions(state, PosixFilePermissions.fromString("rwx------"));
    System.setProperty("lucene.acceleration.native.statePath", state.toString());
    try {
      checkFaultIsolated(fault, required);
    } finally {
      if (previousState == null) System.clearProperty("lucene.acceleration.native.statePath");
      else System.setProperty("lucene.acceleration.native.statePath", previousState);
    }
  }

  private void checkFaultIsolated(int fault, boolean required) throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            config(
                required ? AccelerationConfig.Mode.REQUIRED : AccelerationConfig.Mode.AUTO,
                faultPack(fault),
                required ? Set.of(Capability.VECTOR_I8) : Set.of(),
                Set.of(),
                1));
    assertTrue(runtime.inventory().rejections().toString(), runtime.supports(Capability.VECTOR_I8));
    long before = runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE);
    try (var scope = runtime.newScope()) {
      int rows = 7777;
      ByteBuffer query = scope.allocate(1);
      ByteBuffer data = scope.allocate(rows);
      ByteBuffer output = scope.allocate(rows * 4L).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer metadata = scope.allocate(256);
      ByteBuffer scratch = scope.allocate(0);
      query.put(0, (byte) 1);
      for (int i = 0; i < rows; i++) data.put(i, (byte) 2);
      parameters.putLong(0, rows).putLong(8, 1).putLong(16, 1);
      var batch =
          new AccelerationProvider.Batch(
              Capability.VECTOR_I8,
              List.of(
                  new AccelerationProvider.Input(query, 1, AccelerationProvider.Encoding.BYTES),
                  new AccelerationProvider.Input(data, rows, AccelerationProvider.Encoding.BYTES)),
              new AccelerationProvider.Output(output, rows),
              parameters,
              scratch,
              metadata);
      if (fault == 3 || required) {
        expectThrows(IOException.class, () -> runtime.execute(batch, rows, rows + 1L));
      } else {
        assertFalse(runtime.execute(batch, rows, rows + 1L));
      }
      var counter = runtime.inventory().counters().get(Capability.VECTOR_I8);
      if (fault == 4) {
        assertFalse(counter.quarantined());
        assertEquals(0L, counter.errors());
        assertTrue(runtime.eligible(Capability.VECTOR_I8, rows));
      } else {
        assertTrue(counter.quarantined());
        assertEquals(1L, counter.errors());
        if (required) {
          assertFalse(runtime.inventory().ready());
          expectThrows(
              IllegalStateException.class, () -> runtime.eligible(Capability.VECTOR_I8, rows));
        } else {
          assertTrue(runtime.inventory().ready());
          assertFalse(runtime.eligible(Capability.VECTOR_I8, rows));
        }
      }
    }
    assertEquals(
        before, runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  private Path fakePack(boolean injectApi) throws Exception {
    Path root = localPackDirectory("passive-pack");
    try (var jar = new JarOutputStream(Files.newOutputStream(root.resolve("provider.jar")))) {
      jar.putNextEntry(
          new JarEntry(
              injectApi
                  ? "org/apache/lucene/acceleration/AccelerationProvider.class"
                  : "marker.txt"));
      jar.write(1);
      jar.closeEntry();
    }
    Files.write(root.resolve("native.so"), new byte[] {1});
    String manifest =
        String.format(
            java.util.Locale.ROOT,
            """
            {"manifestSchema":1,"packId":"fixture","packVersion":"1",
            "api":{"major":1,"minor":0},"nativeAbi":{"major":1,"minor":0},
            "adapterBuilds":{"lucene":["lucene-10.5.2-bridge-4"]},
            "artifacts":[{"kind":"provider-jar","path":"provider.jar","sha256":"%s"},
            {"kind":"native-library","path":"native.so","target":"%s","sha256":"%s"}],
            "kernels":[{"operation":"vector.score.i8.exact.v1","schemaMajor":1,
            "arithmeticProfile":"lucene-10.5.2-i8-v1","priority":100,"requiredFeatures":[],
            "inputEncoding":"mixed-le","minimumAlignment":1}]}
            """,
            digest(root.resolve("provider.jar")),
            PackLoader.target(),
            digest(root.resolve("native.so")));
    Files.writeString(root.resolve("manifest.json"), manifest);
    trustedFile(root.resolve("manifest.json"));
    trustedFile(root.resolve("provider.jar"));
    trustedFile(root.resolve("native.so"));
    return root;
  }

  private Path faultPack(int variant) throws Exception {
    Path canonical = nativePack();
    Path faultRoot =
        Path.of(
            System.getProperty("tests.acceleration.faultRoot", canonical.getParent().toString()));
    Path library =
        faultRoot.resolve("fault-runtime-" + variant).resolve("liblucene_acceleration.so");
    assumeTrue(
        "optional test-only native fault library absent: " + library, Files.isRegularFile(library));
    Path root = localPackDirectory("fault-pack-" + variant);
    copyPack(canonical, root);
    Path nativeRelative = canonical.relativize(PackLoader.verify(canonical).library());
    Path destination = root.resolve(nativeRelative);
    String oldDigest = digest(destination);
    Files.copy(library, destination, StandardCopyOption.REPLACE_EXISTING);
    trustedFile(destination);
    Path manifest = root.resolve("manifest.json");
    Files.writeString(manifest, Files.readString(manifest).replace(oldDigest, digest(destination)));
    return root;
  }

  /**
   * Native loading needs default-filesystem paths; Lucene still owns the enclosing temp lifetime.
   */
  private Path localPackDirectory(String prefix) throws IOException {
    Path root = TrustedPathFixtures.directory(prefix);
    trustedDirectory(root);
    return root;
  }

  private static void trustedDirectory(Path directory) throws IOException {
    if (Files.getFileStore(directory).supportsFileAttributeView("posix")) {
      Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwxr-xr-x"));
    }
  }

  private static void trustedFile(Path path) throws IOException {
    if (Files.getFileStore(path).supportsFileAttributeView("posix"))
      Files.setPosixFilePermissions(path, PosixFilePermissions.fromString("rw-r--r--"));
  }

  private static void copyPack(Path canonical, Path root) throws IOException {
    try (var paths = Files.walk(canonical)) {
      for (Path source : paths.toList()) {
        Path destination = root.resolve(canonical.relativize(source));
        if (Files.isDirectory(source)) {
          Files.createDirectories(destination);
          trustedDirectory(destination);
        } else {
          Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
          if (Files.getFileStore(destination).supportsFileAttributeView("posix")) {
            Files.setPosixFilePermissions(
                destination, PosixFilePermissions.fromString("rw-r--r--"));
          }
        }
      }
    }
  }

  private static String digest(Path path) throws Exception {
    return HexFormat.of()
        .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)));
  }
}
