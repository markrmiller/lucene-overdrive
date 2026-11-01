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
import java.io.InputStream;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.jar.Manifest;
import java.util.zip.ZipInputStream;

/** Passive bounded verification precedes every provider constructor and native library load. */
final class PackLoader {
  record Verified(
      Path root,
      String id,
      String version,
      int priority,
      Set<Capability> capabilities,
      List<Path> jars,
      Path library,
      String digest,
      Map<Capability, AccelerationProvider.KernelDescriptor> descriptions,
      String minimumOs,
      String minimumLibc) {}

  record Loaded(Verified manifest, AccelerationProvider.Prepared provider, URLClassLoader loader) {}

  private PackLoader() {}

  static List<Path> discover(Path root) throws IOException {
    if (root == null || !Files.exists(root, LinkOption.NOFOLLOW_LINKS)) return List.of();
    safeRoot(root);
    if (Files.exists(root.resolve("manifest.json"), LinkOption.NOFOLLOW_LINKS))
      return List.of(root);
    try (var entries = Files.list(root)) {
      List<Path> directories =
          entries
              .filter(p -> Files.isDirectory(p, LinkOption.NOFOLLOW_LINKS))
              .sorted()
              .limit(33)
              .toList();
      if (directories.size() > 32) throw new IOException("too many packs");
      return directories;
    }
  }

  @SuppressWarnings("unchecked")
  static Verified verify(Path root) throws IOException {
    safeRoot(root);
    Path manifest = safePath(root, "manifest.json");
    if (Files.size(manifest) > 65536) throw new IOException("manifest too large");
    byte[] manifestBytes;
    try (InputStream stream = Files.newInputStream(manifest)) {
      manifestBytes = stream.readNBytes(65537);
    }
    if (manifestBytes.length > 65536) throw new IOException("manifest too large");
    String json =
        StandardCharsets.UTF_8
            .newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(manifestBytes))
            .toString();
    Map<String, Object> data = object(BoundedJson.parse(json));
    optionalKeys(
        data,
        Set.of(
            "manifestSchema",
            "packId",
            "packVersion",
            "api",
            "adapterBuilds",
            "nativeAbi",
            "artifacts",
            "kernels",
            "minimumJdk",
            "experimental",
            "minimumOs",
            "minimumLibc",
            "requiredFields"));
    keys(object(data.get("api")), Set.of("major", "minor"));
    keys(object(data.get("nativeAbi")), Set.of("major", "minor"));
    if (number(data, "manifestSchema") != 1
        || number(object(data.get("api")), "major") != 1
        || number(object(data.get("api")), "minor") < 0
        || number(object(data.get("nativeAbi")), "major") != 1
        || number(object(data.get("nativeAbi")), "minor") < 0)
      throw new IOException("unsupported manifest/API/ABI");
    if (data.containsKey("minimumJdk") && number(data, "minimumJdk") > Runtime.version().feature())
      throw new IOException("unsupported JDK");
    Map<String, Object> builds = object(data.get("adapterBuilds"));
    if (!array(builds.get("lucene")).contains(AccelerationRuntime.BRIDGE_BUILD))
      throw new IOException("Lucene bridge mismatch");
    String id = string(data, "packId");
    String version = string(data, "packVersion");
    String target = target();
    List<Path> jars = new ArrayList<>();
    Path library = null;
    Set<Path> paths = new HashSet<>();
    for (Object entry : array(data.get("artifacts"))) {
      Map<String, Object> artifact = object(entry);
      keys(artifact, Set.of("kind", "path", "sha256", "target"));
      Path path = safePath(root, string(artifact, "path"));
      if (!paths.add(path)) throw new IOException("duplicate artifact");
      String expected = string(artifact, "sha256");
      if (!expected.matches("[0-9a-f]{64}") || !digest(path).equals(expected))
        throw new IOException("artifact checksum mismatch");
      switch (string(artifact, "kind")) {
        case "provider-jar" -> {
          verifyJar(path);
          jars.add(path);
        }
        case "native-library" -> {
          if (target.equals(string(artifact, "target"))) {
            if (library != null) throw new IOException("duplicate target library");
            library = path;
          }
        }
        default -> throw new IOException("unknown artifact kind");
      }
    }
    if (jars.isEmpty() || jars.size() > 8 || library == null)
      throw new IOException("missing target/provider");
    Set<Capability> capabilities = new HashSet<>();
    Map<Capability, AccelerationProvider.KernelDescriptor> descriptions =
        new java.util.EnumMap<>(Capability.class);
    Integer priority = null;
    Set<String> operations = new HashSet<>();
    for (Object entry : array(data.get("kernels"))) {
      Map<String, Object> kernel = object(entry);
      optionalKeys(
          kernel,
          Set.of(
              "operation",
              "schemaMajor",
              "arithmeticProfile",
              "priority",
              "requiredFeatures",
              "inputEncoding",
              "minimumAlignment",
              "maximumElements",
              "maximumDimensions",
              "maximumBytes",
              "scratchFixedBytes",
              "scratchBytesPerElement",
              "scratchBytesPerDimension",
              "stagingFixedBytes",
              "stagingBytesPerElement",
              "stagingBytesPerDimension",
              "threadSafe",
              "testFingerprint",
              "layouts",
              "optional",
              "requiredFields"));
      Capability capability;
      try {
        String operation = string(kernel, "operation");
        if (!operations.add(operation)) throw new IOException("duplicate kernel operation");
        capability = Capability.fromId(operation);
      } catch (IllegalArgumentException e) {
        if (Boolean.TRUE.equals(kernel.get("optional"))) continue;
        throw new IOException("unknown required kernel", e);
      }
      if (!capabilities.add(capability)
          || number(kernel, "schemaMajor") != 1
          || !string(kernel, "arithmeticProfile").equals(capability.arithmeticProfile))
        throw new IOException("unsupported/duplicate kernel contract");
      long priorityValue = number(kernel, "priority");
      if (priorityValue < Integer.MIN_VALUE || priorityValue > Integer.MAX_VALUE)
        throw new IOException("provider priority out of range");
      int p = (int) priorityValue;
      if (priority != null && priority != p) throw new IOException("mixed provider priorities");
      priority = p;
      long alignment = optionalNumber(kernel, "minimumAlignment", 1);
      if (alignment < 1 || alignment > 64 || Long.bitCount(alignment) != 1)
        throw new IOException("unsupported alignment");
      if (kernel.containsKey("inputEncoding")
          && !Set.of("bytes", "u64-le", "i64-le", "i32-le", "f32-le", "mixed-le")
              .contains(string(kernel, "inputEncoding")))
        throw new IOException("unsupported encoding");
      Set<String> features = new HashSet<>();
      for (Object value : array(kernel.get("requiredFeatures"))) {
        if (!(value instanceof String feature)
            || !Set.of("x86.avx2", "os.xsave-ymm", "x86.fma").contains(feature)
            || !features.add(feature)) throw new IOException("unsupported/duplicate CPU feature");
      }
      Set<AccelerationProvider.LayoutKind> layouts = new HashSet<>();
      if (kernel.containsKey("layouts")) {
        for (Object value : array(kernel.get("layouts"))) {
          try {
            layouts.add(AccelerationProvider.LayoutKind.valueOf((String) value));
          } catch (IllegalArgumentException | ClassCastException e) {
            throw new IOException("unsupported layout", e);
          }
        }
      } else layouts.addAll(Set.of(AccelerationProvider.LayoutKind.values()));
      Set<AccelerationProvider.Encoding> encodings =
          switch (kernel.containsKey("inputEncoding")
              ? string(kernel, "inputEncoding")
              : "mixed-le") {
            case "bytes" -> Set.of(AccelerationProvider.Encoding.BYTES);
            case "u64-le", "i64-le" -> Set.of(AccelerationProvider.Encoding.I64);
            case "i32-le" -> Set.of(AccelerationProvider.Encoding.I32);
            case "f32-le" -> Set.of(AccelerationProvider.Encoding.F32);
            default -> Set.of(AccelerationProvider.Encoding.values());
          };
      var fallback = AccelerationProvider.KernelDescriptor.reference(capability);
      boolean threadSafe =
          !kernel.containsKey("threadSafe") || Boolean.TRUE.equals(kernel.get("threadSafe"));
      long maximumDimensions =
          optionalNumber(kernel, "maximumDimensions", fallback.maximumDimensions());
      if (maximumDimensions < 0 || maximumDimensions > Integer.MAX_VALUE)
        throw new IOException("invalid maximum dimensions");
      descriptions.put(
          capability,
          new AccelerationProvider.KernelDescriptor(
              AccelerationProvider.KernelKey.of(capability),
              encodings,
              layouts,
              optionalNumber(kernel, "maximumElements", fallback.maximumElements()),
              (int) maximumDimensions,
              optionalNumber(kernel, "maximumBytes", fallback.maximumBytes()),
              (int) alignment,
              features,
              new AccelerationProvider.Workspace(
                  optionalNumber(kernel, "scratchFixedBytes", 0),
                  optionalNumber(kernel, "scratchBytesPerElement", 0),
                  optionalNumber(kernel, "scratchBytesPerDimension", 0)),
              new AccelerationProvider.Workspace(
                  optionalNumber(kernel, "stagingFixedBytes", 0),
                  optionalNumber(kernel, "stagingBytesPerElement", 0),
                  optionalNumber(kernel, "stagingBytesPerDimension", 0)),
              threadSafe,
              kernel.containsKey("testFingerprint")
                  ? string(kernel, "testFingerprint")
                  : fallback.testFingerprint()));
    }
    if (capabilities.isEmpty()) throw new IOException("empty pack");
    return new Verified(
        root,
        id,
        version,
        priority == null ? 0 : priority,
        Set.copyOf(capabilities),
        List.copyOf(jars),
        library,
        HexFormat.of().formatHex(sha256().digest(manifestBytes)),
        Map.copyOf(descriptions),
        data.containsKey("minimumOs") ? string(data, "minimumOs") : "",
        data.containsKey("minimumLibc") ? string(data, "minimumLibc") : "");
  }

  // Verification and loading must address the same immutable bytes. Reopening a previously
  // verified user path allows replacement between the checksum and provider/native execution.
  static Verified snapshot(Verified original) throws IOException {
    Path root = snapshotDirectory();
    boolean success = false;
    try {
      Path sourceManifest = safePath(original.root(), "manifest.json");
      byte[] bytes;
      try (var stream = Files.newInputStream(sourceManifest)) {
        bytes = stream.readNBytes(65537);
      }
      if (bytes.length > 65536
          || !HexFormat.of().formatHex(sha256().digest(bytes)).equals(original.digest()))
        throw new IOException("manifest changed after verification");
      Map<String, Object> data =
          object(BoundedJson.parse(new String(bytes, StandardCharsets.UTF_8)));
      Files.write(root.resolve("manifest.json"), bytes);
      Files.setPosixFilePermissions(
          root.resolve("manifest.json"),
          java.nio.file.attribute.PosixFilePermissions.fromString("rw-------"));
      for (Object item : array(data.get("artifacts"))) {
        Map<String, Object> artifact = object(item);
        String relative = string(artifact, "path");
        Path source = safePath(original.root(), relative);
        Path destination = root.resolve(relative);
        Files.createDirectories(
            destination.getParent(),
            java.nio.file.attribute.PosixFilePermissions.asFileAttribute(
                java.nio.file.attribute.PosixFilePermissions.fromString("rwx------")));
        Files.copy(source, destination);
        Files.setPosixFilePermissions(
            destination, java.nio.file.attribute.PosixFilePermissions.fromString("rw-------"));
      }
      Verified result = verify(root);
      if (!result.digest().equals(original.digest()))
        throw new IOException("snapshot identity changed");
      success = true;
      return result;
    } finally {
      if (!success) deleteSnapshot(root);
    }
  }

  static Path snapshotDirectory() throws IOException {
    Path fallback = Path.of("/tmp");
    IOException configuredFailure;
    try {
      return createSnapshotDirectory(Path.of(System.getProperty("java.io.tmpdir", "/tmp")));
    } catch (java.nio.file.InvalidPathException invalid) {
      configuredFailure = new IOException("invalid snapshot temporary parent", invalid);
    } catch (IOException failure) {
      configuredFailure = failure;
    }
    try {
      return createSnapshotDirectory(fallback);
    } catch (IOException failure) {
      failure.addSuppressed(configuredFailure);
      throw failure;
    }
  }

  private static Path createSnapshotDirectory(Path parent) throws IOException {
    // Validate before creating anything: shared writable build trees are not runtime installations.
    TrustedPaths.verifyTemporaryParent(parent);
    Path root =
        Files.createTempDirectory(
            parent,
            "lucene-verified-pack-",
            java.nio.file.attribute.PosixFilePermissions.asFileAttribute(
                java.nio.file.attribute.PosixFilePermissions.fromString("rwx------")));
    try {
      TrustedPaths.verify(root);
      return root;
    } catch (IOException | RuntimeException | Error failure) {
      try {
        Files.delete(root);
      } catch (IOException cleanup) {
        failure.addSuppressed(cleanup);
      }
      throw failure;
    }
  }

  static void deleteSnapshot(Path root) throws IOException {
    try (var paths = Files.walk(root)) {
      for (Path path : paths.sorted(java.util.Comparator.reverseOrder()).toList())
        Files.delete(path);
    }
  }

  private static long optionalNumber(Map<String, Object> object, String name, long fallback)
      throws IOException {
    return object.containsKey(name) ? number(object, name) : fallback;
  }

  static Loaded instantiate(
      Verified manifest,
      NativeBridge bridge,
      org.apache.lucene.acceleration.memory.MemoryLedger ledger)
      throws IOException {
    return instantiate(manifest, bridge, ledger, AccelerationProvider.class.getClassLoader());
  }

  static Loaded instantiate(
      Verified original,
      NativeBridge bridge,
      org.apache.lucene.acceleration.memory.MemoryLedger ledger,
      ClassLoader startupLoader)
      throws IOException {
    Verified manifest = snapshot(original);
    try {
      bridge.verifyPlatform(manifest.minimumOs(), manifest.minimumLibc());
      NativeLibraryPolicy.verify(manifest.library());
      try {
        if (Class.forName(AccelerationProvider.class.getName(), false, startupLoader)
            != AccelerationProvider.class)
          throw new IOException("startup loader has a different API definition");
      } catch (ClassNotFoundException e) {
        throw new IOException("startup loader cannot resolve the host API", e);
      }
    } catch (IOException | RuntimeException | Error failure) {
      try {
        deleteSnapshot(manifest.root());
      } catch (IOException cleanup) {
        failure.addSuppressed(cleanup);
      }
      throw failure;
    }
    var urls =
        manifest.jars().stream()
            .map(
                p -> {
                  try {
                    return p.toUri().toURL();
                  } catch (java.net.MalformedURLException e) {
                    throw new IllegalArgumentException(e);
                  }
                })
            .toArray(java.net.URL[]::new);
    URLClassLoader loader = new URLClassLoader(urls, startupLoader);
    boolean success = false;
    try {
      List<AccelerationProvider> providers = new ArrayList<>();
      var registrations =
          ServiceLoader.load(AccelerationProvider.class, loader).stream().iterator();
      while (registrations.hasNext()) {
        var registration = registrations.next();
        if (registration.type().getClassLoader() != loader)
          throw new IOException("provider not owned by verified loader");
        providers.add(registration.get());
        if (providers.size() > 1)
          throw new IOException("exactly one provider per manifest required");
      }
      if (providers.size() != 1) throw new IOException("missing provider registration");
      AccelerationProvider provider = providers.getFirst();
      var descriptor = provider.descriptor();
      if (!descriptor.id().equals(manifest.id())
          || !descriptor.version().equals(manifest.version())
          || descriptor.priority() != manifest.priority()
          || !descriptor.capabilities().equals(manifest.capabilities()))
        throw new IOException("provider descriptor mismatch");
      var prepared =
          provider.prepare(
              new AccelerationProvider.HostContext() {
                @Override
                public String bridgeBuild() {
                  return AccelerationRuntime.BRIDGE_BUILD;
                }

                @Override
                public AccelerationProvider.Prepared prepareNative(
                    AccelerationProvider.Descriptor d) throws IOException {
                  if (!d.equals(descriptor))
                    throw new IOException("unverified provider descriptor");
                  var verifiedDescriptor =
                      new AccelerationProvider.Descriptor(
                          d.id(),
                          d.version(),
                          d.priority(),
                          d.capabilities(),
                          manifest.descriptions());
                  return bridge.prepare(manifest.library(), verifiedDescriptor, ledger);
                }
              });
      if (!descriptor.capabilities().containsAll(prepared.capabilities()))
        throw new IOException("unadvertised capability");
      success = true;
      return new Loaded(manifest, prepared, loader);
    } finally {
      if (!success) {
        loader.close();
        deleteSnapshot(manifest.root());
      }
    }
  }

  private static void safeRoot(Path root) throws IOException {
    if (!root.isAbsolute() || !Files.isDirectory(root, LinkOption.NOFOLLOW_LINKS))
      throw new IOException("pack root is not an absolute directory");
    for (Path part = root; part != null; part = part.getParent()) {
      if (Files.isSymbolicLink(part)) throw new IOException("pack symlink");
    }
    TrustedPaths.verify(root);
  }

  private static void privateOwnerWrite(Path path) throws IOException {
    if (Files.getFileStore(path).supportsFileAttributeView("posix")) {
      var permissions = Files.getPosixFilePermissions(path, LinkOption.NOFOLLOW_LINKS);
      if (permissions.contains(PosixFilePermission.GROUP_WRITE)
          || permissions.contains(PosixFilePermission.OTHERS_WRITE))
        throw new IOException("pack path writable by peers");
    }
  }

  private static Path safePath(Path root, String value) throws IOException {
    Path relative = Path.of(value);
    if (relative.isAbsolute()
        || value.isEmpty()
        || !relative.normalize().equals(relative)
        || value.contains("\\")
        || relative.startsWith("..")) throw new IOException("unsafe artifact path");
    Path resolved = root.resolve(relative);
    for (Path p = resolved; !p.equals(root); p = p.getParent()) {
      if (Files.isSymbolicLink(p)) throw new IOException("artifact symlink");
      privateOwnerWrite(p);
    }
    if (!Files.isRegularFile(resolved, LinkOption.NOFOLLOW_LINKS))
      throw new IOException("missing artifact");
    if (Files.size(resolved) > 256L * 1024 * 1024) throw new IOException("artifact too large");
    TrustedPaths.verify(resolved);
    return resolved;
  }

  private static void verifyJar(Path path) throws IOException {
    try (var stream = new ZipInputStream(Files.newInputStream(path))) {
      int count = 0;
      long expanded = 0;
      byte[] block = new byte[8192];
      Set<String> names = new HashSet<>();
      for (var entry = stream.getNextEntry(); entry != null; entry = stream.getNextEntry()) {
        String name = entry.getName();
        if (++count > 4096
            || !names.add(name)
            || name.contains("..")
            || name.startsWith("/")
            || !entry.isDirectory()
                && (name.startsWith("org/apache/lucene/acceleration/")
                    || name.contains("/org/apache/lucene/acceleration/"))
                && !name.contains("org/apache/lucene/acceleration/pack/"))
          throw new IOException("provider embeds host API/invalid entry");
        if (name.endsWith("/org/apache/lucene/acceleration/AccelerationProvider.class"))
          throw new IOException("duplicate API");
        if (name.equalsIgnoreCase("META-INF/MANIFEST.MF")) {
          byte[] manifest = stream.readNBytes(65537);
          if (manifest.length > 65536) throw new IOException("JAR manifest too large");
          if (new Manifest(new java.io.ByteArrayInputStream(manifest))
                  .getMainAttributes()
                  .getValue("Class-Path")
              != null) throw new IOException("provider JAR declares unverified class path");
          expanded += manifest.length;
        } else {
          long entryBytes = 0;
          for (int n; (n = stream.read(block)) != -1; ) {
            entryBytes += n;
            expanded += n;
            if (entryBytes > 16L * 1024 * 1024 || expanded > 256L * 1024 * 1024)
              throw new IOException("expanded provider JAR too large");
          }
        }
      }
    }
  }

  static String target() throws IOException {
    if (!System.getProperty("os.name").equals("Linux")) throw new IOException("unqualified OS");
    return switch (System.getProperty("os.arch")) {
      case "amd64", "x86_64" -> "linux-x86_64-glibc";
      case "aarch64" -> "linux-aarch64-glibc";
      default -> throw new IOException("unqualified architecture");
    };
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }

  private static String digest(Path path) throws IOException {
    MessageDigest digest = sha256();
    try (InputStream input = Files.newInputStream(path)) {
      byte[] block = new byte[8192];
      for (int n; (n = input.read(block)) >= 0; ) digest.update(block, 0, n);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> object(Object o) throws IOException {
    if (!(o instanceof Map<?, ?>)) throw new IOException("manifest object expected");
    return (Map<String, Object>) o;
  }

  @SuppressWarnings("unchecked")
  private static List<Object> array(Object o) throws IOException {
    if (!(o instanceof List<?>)) throw new IOException("manifest array expected");
    return (List<Object>) o;
  }

  private static String string(Map<String, Object> m, String key) throws IOException {
    if (!(m.get(key) instanceof String value) || value.isEmpty())
      throw new IOException("manifest string expected");
    return value;
  }

  private static long number(Map<String, Object> m, String key) throws IOException {
    if (!(m.get(key) instanceof Long value)) throw new IOException("manifest integer expected");
    return value;
  }

  private static void optionalKeys(Map<String, Object> values, Set<String> known)
      throws IOException {
    if (values.containsKey("requiredFields")) {
      for (Object field : array(values.get("requiredFields"))) {
        if (!(field instanceof String name) || !known.contains(name) || !values.containsKey(name))
          throw new IOException("unsupported required manifest field");
      }
    }
  }

  private static void keys(Map<String, Object> m, Set<String> allowed) throws IOException {
    if (!allowed.containsAll(m.keySet())) throw new IOException("unknown manifest field");
  }
}
