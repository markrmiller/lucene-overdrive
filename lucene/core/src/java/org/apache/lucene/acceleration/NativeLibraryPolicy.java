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
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Passive ELF dependency audit. Pack-local transitive code and search-path injection are rejected.
 */
final class NativeLibraryPolicy {
  private NativeLibraryPolicy() {}

  private record Load(long offset, long address, long size) {}

  static void verify(Path library) throws IOException {
    if (!System.getProperty("os.name").equals("Linux"))
      throw new IOException("unqualified native dependency platform");
    for (String name : List.of("LD_PRELOAD", "LD_AUDIT")) {
      String value = System.getenv(name);
      if (value != null && !value.isEmpty())
        throw new IOException("native loader injection configured: " + name);
    }
    try (SeekableByteChannel channel = Files.newByteChannel(library)) {
      long size = channel.size();
      ByteBuffer header = read(channel, 0, 64, size);
      if (header.getInt(0) != 0x464c457f || header.get(4) != 2 || header.get(5) != 1)
        throw new IOException("native payload is not 64-bit little-endian ELF");
      long phoff = header.getLong(32);
      int entrySize = Short.toUnsignedInt(header.getShort(54));
      int count = Short.toUnsignedInt(header.getShort(56));
      if (entrySize != 56 || count < 1 || count > 128)
        throw new IOException("unsupported ELF program header bounds");
      List<Load> loads = new ArrayList<>();
      long dynamicOffset = -1, dynamicBytes = 0;
      for (int i = 0; i < count; i++) {
        ByteBuffer entry =
            read(
                channel,
                Math.addExact(phoff, Math.multiplyExact((long) i, entrySize)),
                entrySize,
                size);
        int type = entry.getInt(0);
        long offset = entry.getLong(8), address = entry.getLong(16), bytes = entry.getLong(32);
        if (offset < 0 || address < 0 || bytes < 0 || offset > size - bytes)
          throw new IOException("invalid ELF segment bounds");
        if (type == 1) loads.add(new Load(offset, address, bytes));
        if (type == 2) {
          if (dynamicOffset != -1) throw new IOException("duplicate ELF dynamic segment");
          dynamicOffset = offset;
          dynamicBytes = bytes;
        }
      }
      if (dynamicOffset < 0 || dynamicBytes < 16 || dynamicBytes > 65536 || dynamicBytes % 16 != 0)
        throw new IOException("invalid ELF dynamic table");
      ByteBuffer dynamic = read(channel, dynamicOffset, (int) dynamicBytes, size);
      List<Long> needed = new ArrayList<>();
      long stringsAddress = -1, stringsSize = -1;
      boolean terminated = false;
      for (int i = 0; i < dynamicBytes; i += 16) {
        long tag = dynamic.getLong(i), value = dynamic.getLong(i + 8);
        if (tag == 0) {
          terminated = true;
          break;
        }
        if (tag == 1) needed.add(value);
        else if (tag == 5) stringsAddress = value;
        else if (tag == 10) stringsSize = value;
        else if ((tag == 15 || tag == 29) && value != 0)
          throw new IOException("pack declares a native dependency search path");
      }
      if (!terminated
          || needed.size() > 8
          || stringsAddress < 0
          || stringsSize < 1
          || stringsSize > 1 << 20) throw new IOException("invalid ELF string table bounds");
      long stringsOffset = -1;
      for (Load load : loads) {
        if (stringsAddress >= load.address()
            && stringsAddress - load.address() <= load.size() - stringsSize) {
          stringsOffset = Math.addExact(load.offset(), stringsAddress - load.address());
          break;
        }
      }
      if (stringsOffset < 0) throw new IOException("ELF strings outside a file-backed segment");
      ByteBuffer strings = read(channel, stringsOffset, (int) stringsSize, size);
      Set<String> allowed = Set.of("libc.so.6", "libm.so.6", "libgcc_s.so.1");
      for (long offset : needed) {
        if (offset < 0 || offset >= stringsSize)
          throw new IOException("invalid ELF dependency offset");
        StringBuilder name = new StringBuilder();
        boolean end = false;
        for (long i = offset; i < stringsSize && name.length() <= 128; i++) {
          int ch = Byte.toUnsignedInt(strings.get((int) i));
          if (ch == 0) {
            end = true;
            break;
          }
          if (ch < 32 || ch > 126) throw new IOException("invalid ELF dependency name");
          name.append((char) ch);
        }
        if (!end || !allowed.contains(name.toString()))
          throw new IOException("unapproved transitive native dependency");
        verifySearchResolution(name.toString());
      }
    } catch (ArithmeticException e) {
      throw new IOException("ELF offset arithmetic overflow", e);
    }
  }

  private static void verifySearchResolution(String dependency) throws IOException {
    String search = System.getenv("LD_LIBRARY_PATH");
    if (search == null || search.isEmpty()) return;
    for (String directory : search.split(":", -1)) {
      Path candidate = Path.of(directory.isEmpty() ? "." : directory).resolve(dependency);
      if (!Files.exists(candidate)) continue;
      Path actual = candidate.toRealPath();
      if (!Files.isRegularFile(actual)
          || !actual.startsWith("/usr/lib") && !actual.startsWith("/lib"))
        throw new IOException("native dependency shadowed by non-system search entry");
    }
  }

  private static ByteBuffer read(SeekableByteChannel channel, long offset, int bytes, long size)
      throws IOException {
    if (offset < 0 || bytes < 0 || offset > size - bytes)
      throw new IOException("truncated ELF metadata");
    ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    channel.position(offset);
    while (buffer.hasRemaining()) {
      if (channel.read(buffer) < 0) throw new IOException("truncated ELF metadata");
    }
    return buffer.flip();
  }
}
