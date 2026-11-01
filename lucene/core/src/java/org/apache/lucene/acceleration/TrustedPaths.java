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
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;

/** Linux installation trust: current-UID/root ownership and no replaceable peer-writable paths. */
final class TrustedPaths {
  private TrustedPaths() {}

  private static int effectiveUid() throws IOException {
    // Read kernel process credentials; user.name/user.home and proc inode ownership are not UID
    // APIs.
    Path status = Path.of("/proc", Long.toString(ProcessHandle.current().pid()), "status");
    String text;
    try (var stream = Files.newInputStream(status)) {
      text = new String(stream.readNBytes(4096), java.nio.charset.StandardCharsets.US_ASCII);
    }
    for (String line : text.split("\n")) {
      if (!line.startsWith("Uid:")) continue;
      String[] fields = line.substring(4).strip().split("\\s+");
      if (fields.length != 4 || !fields[1].matches("[0-9]{1,10}")) break;
      long uid = Long.parseLong(fields[1]);
      if (uid <= 0xffffffffL) return (int) uid;
      break;
    }
    throw new IOException("effective UID cannot be verified");
  }

  static void verifyTemporaryParent(Path directory) throws IOException {
    if (!Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS))
      throw new IOException("temporary parent is not a directory");
    verify(directory, true);
  }

  static void verify(Path path) throws IOException {
    verify(path, false);
  }

  private static void verify(Path path, boolean temporaryParent) throws IOException {
    if (!path.isAbsolute() || !path.normalize().equals(path))
      throw new IOException("trusted path must be absolute and normalized");
    int currentUid = effectiveUid();
    for (Path part = path; part != null; part = part.getParent()) {
      if (Files.isSymbolicLink(part)) throw new IOException("untrusted path symlink");
      var attributes = Files.readAttributes(part, "unix:uid,mode", LinkOption.NOFOLLOW_LINKS);
      int uid = (int) attributes.get("uid"), mode = (int) attributes.get("mode");
      if (uid != 0 && uid != currentUid) throw new IOException("path owned by another UID");
      // Root-owned sticky directories (notably /tmp) protect each child's directory entry.
      boolean stickyRootDirectory =
          uid == 0
              && (mode & 0170000) == 0040000
              && (mode & 01000) != 0
              && (temporaryParent || !part.equals(path));
      if ((mode & 0022) != 0 && !stickyRootDirectory)
        throw new IOException(
            part.equals(path) ? "path writable by peers" : "ancestor writable by peers");
    }
  }
}
