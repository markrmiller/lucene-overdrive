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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Drop-in installation defaults retain bounded capacity and explicit operator overrides. */
public class TestClasspathPack extends LuceneTestCase {
  public void testBundledPolicyRegistersNativeKernelWithoutPackPath() throws Exception {
    String bundle = System.getProperty("tests.classpath.pack");
    assumeTrue("optional classpath reference pack not supplied", bundle != null);
    Path jar = TrustedPathFixtures.directory("classpath-native").resolve("nitrous.jar");
    Files.copy(Path.of(bundle), jar);
    try (var loader =
            new URLClassLoader(
                new java.net.URL[] {jar.toUri().toURL()},
                AccelerationProvider.class.getClassLoader());
        var runtime =
            new AccelerationRuntime(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO, null, java.util.Set.of(), java.util.Set.of(), 1),
                loader)) {
      assertEquals(
          runtime.inventory().rejections().toString(),
          AccelerationRuntime.State.FROZEN,
          runtime.inventory().state());
      assertTrue(runtime.supports(Capability.DOCVALUES_RANGE_FILTER));
      assertFalse(
          "bundle must not enroll unmeasured operations", runtime.supports(Capability.VECTOR_I8));
      try (var scope = runtime.newScope()) {
        int rows = 12500, words = (rows + 63) / 64;
        var values = scope.allocate((long) rows * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        var selected = scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        var output = scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        var parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < rows; i++) values.putLong(i * Long.BYTES, i - rows / 2);
        for (int i = 0; i < words; i++) selected.putLong(i * Long.BYTES, -1L);
        selected.putLong((words - 1) * Long.BYTES, (1L << (rows & 63)) - 1);
        parameters.putLong(0, rows).putLong(24, -400).putLong(32, 400);
        var batch =
            new AccelerationProvider.Batch(
                Capability.DOCVALUES_RANGE_FILTER,
                List.of(
                    new AccelerationProvider.Input(values, rows, AccelerationProvider.Encoding.I64),
                    new AccelerationProvider.Input(
                        selected, words, AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(output, words),
                parameters,
                scope.allocate(0),
                scope.allocate(256));
        assertFalse("scoped policy must reject generic calls", runtime.execute(batch, rows, 0));
        assertTrue(runtime.execute(batch, rows, 0, null, "solr.frange.full-live.v1"));
        for (int i = 0; i < rows; i++) {
          boolean expected = i - rows / 2 >= -400 && i - rows / 2 <= 400;
          assertEquals(expected, (output.getLong((i >>> 6) * Long.BYTES) & (1L << (i & 63))) != 0);
        }
        assertEquals(
            1, runtime.inventory().counters().get(Capability.DOCVALUES_RANGE_FILTER).nativeCalls());
      }
    }
  }

  public void testPassiveExtractionAndCleanup() throws Exception {
    Path jar =
        bundle(
            Map.of(
                "manifest.json",
                new byte[] {'{', '}'},
                "qualification.tsv",
                new byte[0],
                "native/library.so",
                new byte[] {1, 2, 3}));
    Path root;
    try (var loader = loader(jar);
        var packs = ClasspathPacks.discover(loader)) {
      assertEquals(1, packs.roots().size());
      root = packs.roots().getFirst();
      assertArrayEquals(
          new byte[] {1, 2, 3}, Files.readAllBytes(root.resolve("native/library.so")));
    }
    assertFalse(Files.exists(root));
  }

  public void testUnsafeAndExcessiveZipEntries() throws Exception {
    for (String name :
        List.of("../escape", "/absolute", "native/../escape", "native\\escape", "x".repeat(2049))) {
      Path jar = bundle(Map.of(name, new byte[] {1}));
      expectThrows(IOException.class, () -> ClasspathPacks.unpack(resource(jar)));
    }
    Map<String, byte[]> entries = new LinkedHashMap<>();
    for (int i = 0; i < 129; i++) entries.put("file" + i, new byte[0]);
    Path excessive = bundle(entries);
    expectThrows(IOException.class, () -> ClasspathPacks.unpack(resource(excessive)));
    Path missingPolicy = bundle(Map.of("manifest.json", new byte[] {'{', '}'}));
    expectThrows(IOException.class, () -> ClasspathPacks.unpack(resource(missingPolicy)));
  }

  public void testCompressedEntryCannotExceedExpandedLimit() throws Exception {
    var compressed = new ByteArrayOutputStream();
    try (var inner = new ZipOutputStream(compressed)) {
      inner.putNextEntry(new ZipEntry("native/oversized.so"));
      byte[] block = new byte[8192];
      for (long written = 0; written <= ClasspathPacks.MAXIMUM_ENTRY_BYTES; written += block.length)
        inner.write(block);
      inner.closeEntry();
    }
    Path jar = TrustedPathFixtures.directory("classpath-compressed").resolve("nitrous.jar");
    try (var outer = new JarOutputStream(Files.newOutputStream(jar))) {
      outer.putNextEntry(new JarEntry(ClasspathPacks.RESOURCE));
      outer.write(compressed.toByteArray());
      outer.closeEntry();
    }
    Files.setPosixFilePermissions(
        jar, java.nio.file.attribute.PosixFilePermissions.fromString("rw-------"));
    assertTrue(
        "the small compressed container must not admit oversized native content",
        Files.size(jar) < 1 << 20);
    assertTrue(
        expectThrows(IOException.class, () -> ClasspathPacks.unpack(resource(jar)))
            .getMessage()
            .contains("too large"));
  }

  public void testMalformedClasspathPackFailsSoftBeforeNativeStartup() throws Exception {
    Path jar =
        bundle(Map.of("manifest.json", new byte[] {'{', '}'}, "qualification.tsv", new byte[0]));
    try (var loader = loader(jar);
        var runtime =
            new AccelerationRuntime(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO, null, Set.of(), Set.of(), 1),
                loader)) {
      assertEquals(AccelerationRuntime.State.JAVA_ONLY, runtime.inventory().state());
      assertFalse(runtime.hasKernels());
      assertTrue(runtime.inventory().ready());
      assertTrue(
          runtime.inventory().rejections().toString(),
          runtime.inventory().rejections().stream()
              .anyMatch(rejection -> rejection.startsWith("CLASSPATH_VERIFY:")));
    }
  }

  public void testRemoteResourceIsRejectedWithoutConnecting() throws Exception {
    var remote =
        URI.create("jar:http://127.0.0.1:9/nitrous.jar!/" + ClasspathPacks.RESOURCE).toURL();
    assertTrue(
        expectThrows(IOException.class, () -> ClasspathPacks.unpack(remote))
            .getMessage()
            .contains("local JAR"));
  }

  public void testUntrustedJarAndSymlinkAreRejected() throws Exception {
    Path jar =
        bundle(Map.of("manifest.json", new byte[] {'{', '}'}, "qualification.tsv", new byte[0]));
    Path alias = jar.resolveSibling("alias.jar");
    Files.createSymbolicLink(alias, jar);
    expectThrows(IOException.class, () -> ClasspathPacks.unpack(resource(alias)));
    Files.setPosixFilePermissions(
        jar, java.nio.file.attribute.PosixFilePermissions.fromString("rw-rw-rw-"));
    expectThrows(IOException.class, () -> ClasspathPacks.unpack(resource(jar)));
  }

  public void testOffZeroBudgetsAndExplicitPathDoNotDiscoverClasspath() throws Exception {
    ClassLoader untouched =
        new ClassLoader(AccelerationProvider.class.getClassLoader()) {
          @Override
          public Enumeration<URL> getResources(String name) {
            throw new AssertionError("disabled or overridden discovery touched " + name);
          }
        };
    var configs =
        List.of(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.OFF, null, Set.of(), Set.of(), 1),
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO,
                TrustedPathFixtures.directory("explicit").resolve("absent"),
                Set.of(),
                Set.of(),
                1),
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                null,
                Set.of(),
                Set.of(),
                0,
                0,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT),
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                null,
                Set.of(),
                Set.of(),
                16L << 20,
                8L << 20,
                1,
                false,
                AccelerationConfig.StoragePolicy.CURRENT));
    for (var config : configs) {
      try (var runtime = new AccelerationRuntime(config, untouched)) {
        assertEquals(AccelerationRuntime.State.JAVA_ONLY, runtime.inventory().state());
      }
    }
  }

  public void testAbsentPackKeepsJavaWithoutDiagnostics() throws Exception {
    ClassLoader absent =
        new ClassLoader(AccelerationProvider.class.getClassLoader()) {
          @Override
          public Enumeration<URL> getResources(String name) {
            return Collections.emptyEnumeration();
          }
        };
    try (var runtime =
        new AccelerationRuntime(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO, null, Set.of(), Set.of(), 1),
            absent)) {
      assertEquals(AccelerationRuntime.State.JAVA_ONLY, runtime.inventory().state());
      assertTrue(runtime.inventory().rejections().isEmpty());
    }
  }

  public void testEmptyOrInvalidExternalPolicyDoesNotRestoreBundledEnrollment() throws Exception {
    String source = System.getProperty("tests.classpath.pack");
    assumeTrue("optional classpath reference pack not supplied", source != null);
    Path jar = TrustedPathFixtures.directory("classpath-override").resolve("nitrous.jar");
    Files.copy(Path.of(source), jar);
    String key = "lucene.acceleration.native.qualificationPath", previous = System.getProperty(key);
    try (var loader = loader(jar)) {
      for (String override : List.of("", jar.resolveSibling("missing-policy.tsv").toString())) {
        System.setProperty(key, override);
        try (var runtime =
            new AccelerationRuntime(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO, null, Set.of(), Set.of(), 1),
                loader)) {
          assertEquals(
              runtime.inventory().rejections().toString(),
              AccelerationRuntime.State.JAVA_ONLY,
              runtime.inventory().state());
          assertFalse(runtime.supports(Capability.DOCVALUES_RANGE_FILTER));
        }
      }
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
    }
  }

  private static URLClassLoader loader(Path jar) throws Exception {
    return new URLClassLoader(
        new URL[] {jar.toUri().toURL()}, AccelerationProvider.class.getClassLoader());
  }

  private static URL resource(Path jar) throws Exception {
    return URI.create("jar:" + jar.toUri().toURL() + "!/" + ClasspathPacks.RESOURCE).toURL();
  }

  private static Path bundle(Map<String, byte[]> entries) throws Exception {
    var bytes = new ByteArrayOutputStream();
    try (var zip = new ZipOutputStream(bytes)) {
      for (var entry : entries.entrySet()) {
        zip.putNextEntry(new ZipEntry(entry.getKey()));
        zip.write(entry.getValue());
        zip.closeEntry();
      }
    }
    Path jar = TrustedPathFixtures.directory("classpath-passive").resolve("nitrous.jar");
    try (var outer = new JarOutputStream(Files.newOutputStream(jar))) {
      outer.putNextEntry(new JarEntry(ClasspathPacks.RESOURCE));
      outer.write(bytes.toByteArray());
      outer.closeEntry();
    }
    Files.setPosixFilePermissions(
        jar, java.nio.file.attribute.PosixFilePermissions.fromString("rw-------"));
    return jar;
  }

  public void testClasspathInstallationNeedsNoBudgetOrExperimentalProperties() {
    TrustedPathFixtures.assumeHostAccess();
    String[] keys = {
      "lucene.acceleration.memory.nativeBytes",
      "lucene.acceleration.memory.requestBytes",
      "lucene.acceleration.experimental.enabled"
    };
    String[] previous = new String[keys.length];
    try {
      for (int i = 0; i < keys.length; i++) {
        previous[i] = System.getProperty(keys[i]);
        System.clearProperty(keys[i]);
      }
      var config = AccelerationConfig.fromSystemProperties();
      assertTrue("drop-in capacity must be usable by default", config.nativeBytes() > 0);
      assertTrue(config.nativeBytes() <= 64L << 20);
      assertTrue(config.requestBytes() > 0);
      assertTrue(config.requestBytes() <= 8L << 20);
      assertTrue(config.experimental());
      assertEquals(AccelerationConfig.StoragePolicy.CURRENT, config.storagePolicy());
      System.setProperty(keys[0], "0");
      System.setProperty(keys[1], "0");
      System.setProperty(keys[2], "false");
      config = AccelerationConfig.fromSystemProperties();
      assertEquals(0, config.nativeBytes());
      assertEquals(0, config.requestBytes());
      assertFalse(config.experimental());
    } finally {
      for (int i = 0; i < keys.length; i++) {
        if (previous[i] == null) System.clearProperty(keys[i]);
        else System.setProperty(keys[i], previous[i]);
      }
    }
  }
}
