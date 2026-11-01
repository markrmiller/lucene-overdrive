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

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.zip.ZipInputStream;

/** Passive resource discovery; provider classes remain inside the verified pack loader. */
final class ClasspathPacks implements AutoCloseable {
  static final String RESOURCE = "META-INF/lucene-acceleration/pack.zip";
  private static final long MAXIMUM_JAR_BYTES = 256L << 20;
  static final long MAXIMUM_ENTRY_BYTES = 64L << 20;
  static final long MAXIMUM_EXPANDED_BYTES = 128L << 20;
  private final List<Path> roots;
  private boolean closed;

  private ClasspathPacks(List<Path> roots) {
    this.roots = List.copyOf(roots);
  }

  static ClasspathPacks empty() {
    return new ClasspathPacks(List.of());
  }

  static ClasspathPacks discover(ClassLoader loader) throws IOException {
    List<Path> roots = new ArrayList<>();
    boolean success = false;
    try {
      var resources = loader.getResources(RESOURCE);
      var identities = new HashSet<String>();
      int count = 0;
      while (resources.hasMoreElements()) {
        URL resource = resources.nextElement();
        if (++count > 32) throw new IOException("too many classpath packs");
        if (identities.add(resource.toExternalForm())) roots.add(unpack(resource));
      }
      success = true;
      return new ClasspathPacks(roots);
    } finally {
      if (!success) new ClasspathPacks(roots).close();
    }
  }

  List<Path> roots() {
    return roots;
  }

  static Path unpack(URL resource) throws IOException {
    if (!resource.getProtocol().equals("jar"))
      throw new IOException("classpath pack must be a local JAR");
    var connection = (JarURLConnection) resource.openConnection();
    connection.setUseCaches(false);
    if (!connection.getJarFileURL().getProtocol().equals("file")
        || !RESOURCE.equals(connection.getEntryName()))
      throw new IOException("classpath pack must be a local JAR resource");
    Path source;
    try {
      source = Path.of(connection.getJarFileURL().toURI());
    } catch (URISyntaxException | IllegalArgumentException invalid) {
      throw new IOException("invalid classpath pack path", invalid);
    }
    TrustedPaths.verify(source);
    if (!Files.isRegularFile(source, LinkOption.NOFOLLOW_LINKS)
        || Files.size(source) > MAXIMUM_JAR_BYTES)
      throw new IOException("invalid classpath pack JAR");
    Path root = PackLoader.snapshotDirectory();
    boolean success = false;
    try (var jar = new ZipInputStream(Files.newInputStream(source))) {
      var names = new HashSet<String>();
      boolean found = false;
      long expanded = 0;
      byte[] block = new byte[8192];
      for (var entry = jar.getNextEntry(); entry != null; entry = jar.getNextEntry()) {
        if (entry.getName().length() > 2048 || !names.add(entry.getName()) || names.size() > 4096)
          throw new IOException("duplicate or excessive classpath JAR entries");
        var bounded = new ResourceInput(jar, MAXIMUM_JAR_BYTES - expanded);
        if (entry.getName().equals(RESOURCE)) {
          found = true;
          try (var stream = new ZipInputStream(bounded)) {
            extract(stream, root);
          }
        }
        // Consume the complete outer entry, including any inner ZIP directory/trailing bytes.
        while (bounded.read(block) != -1) {}
        expanded += bounded.count;
      }
      if (!found) throw new IOException("missing classpath pack");
      if (!Files.isRegularFile(root.resolve("manifest.json"))
          || !Files.isRegularFile(root.resolve("qualification.tsv")))
        throw new IOException("classpath pack missing manifest or qualification");
      success = true;
      return root;
    } finally {
      if (!success) PackLoader.deleteSnapshot(root);
    }
  }

  private static void extract(ZipInputStream stream, Path root) throws IOException {
    byte[] block = new byte[8192];
    var names = new HashSet<String>();
    long expanded = 0;
    for (var entry = stream.getNextEntry(); entry != null; entry = stream.getNextEntry()) {
      String name = entry.getName();
      if (name.length() > 2048) throw new IOException("classpath pack entry name too long");
      Path relative = Path.of(name);
      if (!names.add(name)
          || names.size() > 128
          || name.isEmpty()
          || name.contains("\\")
          || relative.isAbsolute()
          || !relative.normalize().equals(relative)
          || relative.startsWith("..")
          || !relative.toString().equals(name))
        throw new IOException("unsafe classpath pack entry");
      if (entry.isDirectory()) throw new IOException("explicit pack directories unsupported");
      Path output = root.resolve(relative);
      Files.createDirectories(
          output.getParent(),
          PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
      long entryBytes = 0;
      try (var destination =
          Files.newOutputStream(
              Files.createFile(
                  output,
                  PosixFilePermissions.asFileAttribute(
                      PosixFilePermissions.fromString("rw-------"))))) {
        for (int n; (n = stream.read(block)) != -1; ) {
          expanded += n;
          entryBytes += n;
          if (expanded > MAXIMUM_EXPANDED_BYTES || entryBytes > MAXIMUM_ENTRY_BYTES)
            throw new IOException("expanded classpath pack too large");
          destination.write(block, 0, n);
        }
      }
    }
  }

  /** Keeps inner ZIP closure from closing its parent and bounds all container bytes consumed. */
  private static final class ResourceInput extends FilterInputStream {
    private final long maximum;
    private long count;

    ResourceInput(InputStream input, long maximum) {
      super(input);
      this.maximum = maximum;
    }

    private void consumed(int bytes) throws IOException {
      if (bytes > 0 && (count += bytes) > maximum)
        throw new IOException("expanded classpath JAR too large");
    }

    @Override
    public int read() throws IOException {
      int value = in.read();
      consumed(value < 0 ? 0 : 1);
      return value;
    }

    @Override
    public int read(byte[] bytes, int offset, int length) throws IOException {
      int size = in.read(bytes, offset, length);
      consumed(size);
      return size;
    }

    @Override
    public void close() {
      // The enclosing outer ZIP owns this stream.
    }
  }

  @Override
  public void close() throws IOException {
    if (closed) return;
    closed = true;
    IOException failure = null;
    for (Path root : roots) {
      try {
        PackLoader.deleteSnapshot(root);
      } catch (IOException error) {
        if (failure == null) failure = error;
        else failure.addSuppressed(error);
      }
    }
    if (failure != null) throw failure;
  }
}
