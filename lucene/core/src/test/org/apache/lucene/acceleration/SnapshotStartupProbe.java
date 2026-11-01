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

import java.nio.file.Files;
import java.nio.file.Path;

/** Forked startup boundary: initialize Java's default temp location from the supplied JVM flag. */
public final class SnapshotStartupProbe {
  public static void main(String[] args) throws Exception {
    var original = PackLoader.verify(Path.of(args[0]));
    Path snapshotRoot;
    var snapshot = PackLoader.snapshot(original);
    snapshotRoot = snapshot.root();
    try {
      if (!snapshot.digest().equals(original.digest())) throw new AssertionError("snapshot digest");
      if (!snapshot.root().getParent().equals(Path.of("/tmp")))
        throw new AssertionError("unsafe tmpdir was used: " + snapshot.root());
      if (!Files.getPosixFilePermissions(snapshot.root())
          .equals(java.nio.file.attribute.PosixFilePermissions.fromString("rwx------")))
        throw new AssertionError("snapshot directory is not private");
      if (!java.util.Arrays.equals(
          Files.readAllBytes(original.library()), Files.readAllBytes(snapshot.library())))
        throw new AssertionError("snapshot payload");
      System.out.println("SNAPSHOT_OK:" + snapshot.digest());
    } finally {
      PackLoader.deleteSnapshot(snapshotRoot);
    }
    if (Files.exists(snapshotRoot)) throw new AssertionError("snapshot cleanup");
  }
}
